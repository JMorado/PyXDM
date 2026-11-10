"""Tests for the CLI entry point (cli.py)."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from ..cli import main

DATA_DIR = Path(__file__).parent / "data"
MOLDEN_FILE = str(DATA_DIR / "water.orca.molden.input")


def run_cli(monkeypatch, args: list[str]) -> None:
    """Patch sys.argv and invoke main()."""
    monkeypatch.setattr(sys, "argv", ["pyxdm"] + args)
    main()


class TestCLI:
    def test_runs_without_error(self, monkeypatch, tmp_path):
        out = tmp_path / "water.h5"
        run_cli(monkeypatch, [MOLDEN_FILE, "--scheme", "mbis", "-o", str(out)])

    def test_output_file_created(self, monkeypatch, tmp_path):
        out = tmp_path / "water.h5"
        run_cli(monkeypatch, [MOLDEN_FILE, "--scheme", "mbis", "-o", str(out)])
        assert out.exists()

    def test_output_has_expected_groups(self, monkeypatch, tmp_path):
        out = tmp_path / "water.h5"
        run_cli(monkeypatch, [MOLDEN_FILE, "--scheme", "mbis", "-o", str(out)])
        with h5py.File(out, "r") as f:
            assert "metadata" in f
            assert "molecule" in f
            assert "mbis" in f

    def test_molecule_metadata(self, monkeypatch, tmp_path):
        out = tmp_path / "water.h5"
        run_cli(monkeypatch, [MOLDEN_FILE, "--scheme", "mbis", "-o", str(out)])
        with h5py.File(out, "r") as f:
            symbols = [s.decode() for s in f["molecule/atomic_symbols"][:]]
            assert symbols == ["O", "H", "H"]
            assert f["molecule"].attrs["natom"] == 3

    def test_xdm_moments_vs_reference(self, monkeypatch, tmp_path, water_ref_mbis):
        out = tmp_path / "water.h5"
        run_cli(monkeypatch, [MOLDEN_FILE, "--scheme", "mbis", "-o", str(out)])
        ref_map = {"<M1^2>": "M1_sq", "<M2^2>": "M2_sq", "<M3^2>": "M3_sq"}
        with h5py.File(out, "r") as f:
            for h5_key, ref_key in ref_map.items():
                stored = f[f"mbis/xdm/mbis/xdm_results/{h5_key}"][:]
                np.testing.assert_allclose(
                    stored,
                    water_ref_mbis[ref_key],
                    rtol=1e-2,
                    err_msg=f"{h5_key} from CLI output doesn't match postg reference",
                )

    def test_invalid_scheme_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [MOLDEN_FILE, "--scheme", "invalid_scheme"])
        with pytest.raises(SystemExit):
            main()


def _write_mesh_and_weights(tmp_path: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    """Write a water mesh and its Becke weights as text files."""
    import horton as ht
    from horton.part import BeckeWPart

    mol = ht.IOData.from_file(MOLDEN_FILE)
    grid = ht.BeckeMolGrid(mol.coordinates, mol.numbers, mol.pseudo_numbers, agspec="coarse", mode="keep", random_rotate=False)
    density = mol.obasis.compute_grid_density_dm(mol.get_dm_full(), grid.points)
    part = BeckeWPart(mol.coordinates, mol.numbers, mol.pseudo_numbers, grid, density, local=False)
    part.do_all()
    weights = np.column_stack([part.cache.load("at_weights", i) for i in range(mol.natom)])

    mesh_file, weights_file = tmp_path / "mesh.txt", tmp_path / "weights.txt"
    np.savetxt(mesh_file, np.column_stack([grid.points, grid.weights]), header="x y z w")
    np.savetxt(weights_file, weights, header="weights")
    return mesh_file, weights_file, np.column_stack([grid.points, grid.weights]), weights


class TestExternalWeights:
    def test_moments_match_direct_calculation(self, monkeypatch, tmp_path):
        """--mesh with --weights gives the moments of the same weights computed directly."""
        from ..core import XDMCalculator
        from ..grids import CustomGrid
        from ..partitioning.partitioning import ExternalPartition

        mesh_file, weights_file, mesh, weights = _write_mesh_and_weights(tmp_path)
        out = tmp_path / "water.h5"
        run_cli(monkeypatch, [MOLDEN_FILE, "--mesh", str(mesh_file), "--weights", str(weights_file), "-o", str(out)])

        import horton as ht

        mol = ht.IOData.from_file(MOLDEN_FILE)
        grid = CustomGrid(np.ascontiguousarray(mesh[:, :3]), mesh[:, 3])
        expected = XDMCalculator(mol).calculate_xdm_moments(ExternalPartition(grid, weights), grid, order=[1, 2, 3])
        with h5py.File(out, "r") as f:
            for key in ("<M1^2>", "<M2^2>", "<M3^2>"):
                np.testing.assert_allclose(
                    f[f"external/xdm/external/xdm_results/{key}"][:], expected["external"]["xdm_results"][key], rtol=1e-6
                )

    def test_weights_require_mesh(self, monkeypatch, tmp_path):
        """--weights without --mesh is rejected."""
        _, weights_file, _, _ = _write_mesh_and_weights(tmp_path)
        with pytest.raises(SystemExit):
            run_cli(monkeypatch, [MOLDEN_FILE, "--weights", str(weights_file)])
