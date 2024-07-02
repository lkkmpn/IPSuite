import logging

import ase
import matplotlib.pyplot as plt
import numpy as np
import zntrack
from ase.cell import Cell
from numpy.random import default_rng

from ipsuite import analysis, base

log = logging.getLogger(__name__)


class SurfaceRasterScan(base.ProcessSingleAtom):
    """This class generates periodic structures by creating a vacuum slab in the
    z-direction and adding additives at various positions. It is useful for generating
    input structures for surface training simulations or in combination with the
    SurfaceRasterMetrics class to analyze how well surface interactions are captured
    in the training.

    Attributes
    ----------
    symbol: str
        ASE symbol representing the additives.
    z_dist_list: list[float]
         A list of z-distances at which additives will be added.
    n_conf_per_dist: list[int]
        The number of configurations to generate per z-distance.
    cell_fraction: list[float]
        Fractional scaling of the unit cell in x and y directions.
    random: bool
       If True, additives are placed randomly within the specified cell_fraction.
    max_deflection_shift: float
        Maximum random displacement for each atom.
    seed: int
        Seed for randomly distributing the additive.
    """

    symbol: str = zntrack.params()
    z_dist_list: list[float] = zntrack.params()
    n_conf_per_dist: list[int] = zntrack.params([5, 5])
    cell_fraction: list[float] = zntrack.params([1, 1])
    max_deflection_shift: float = zntrack.params(None)
    seed: int = zntrack.params(1)
    cell_displacement: list[float] = zntrack.params([0, 0])
    
    def run(self) -> None:
        rng = default_rng(self.seed)

        atoms = self.get_data()

        cell = atoms.cell
        cellpar = cell.cellpar()
        cell = np.array(cell)

        z_max = max(atoms.get_positions()[:, 2])

        if not isinstance(self.n_conf_per_dist, list):
            self.n_conf_per_dist = [self.n_conf_per_dist, self.n_conf_per_dist]
        if not isinstance(self.cell_fraction, list):
            self.cell_fraction = [self.cell_fraction, self.cell_fraction]
        atoms_list = []
        for z_dist in self.z_dist_list:
            if cellpar[2] < z_max + z_dist + 10:
                cellpar[2] = z_max + z_dist + 10
                new_cell = Cell.fromcellpar(cellpar)
                atoms.set_cell(new_cell)
                log.warning("vacuum was extended")

            fraction = [(2*(self.n_conf_per_dist[0]))**-1, (2*(self.n_conf_per_dist[1]))**-1]
            a_scaling = np.linspace(fraction[0], 1-fraction[0], self.n_conf_per_dist[0])
            b_scaling = np.linspace(fraction[1], 1-fraction[1], self.n_conf_per_dist[1])

            a_vec = cell[0, :2] * self.cell_fraction[0]
            scaled_a_vecs = a_scaling[:, np.newaxis] * a_vec
            b_vec = cell[1, :2] * self.cell_fraction[1]
            scaled_b_vecs = b_scaling[:, np.newaxis] * b_vec

            for a in scaled_a_vecs:
                for b in scaled_b_vecs:
                    if self.max_deflection_shift is not None:
                        new_atoms = atoms.copy()
                        deflection = rng.uniform(
                            -self.max_deflection_shift,
                            self.max_deflection_shift,
                            size=new_atoms.positions.shape,
                        )
                        new_atoms.positions += deflection
                        atoms_list.append(new_atoms)
                    else:
                        atoms_list.append(atoms.copy())

                    cart_pos = a + b
                    extension = ase.Atoms(
                        self.symbol, [[cart_pos[0] + self.cell_displacement[0], cart_pos[1] + self.cell_displacement[1], z_max + z_dist]]
                    )
                    atoms_list[-1].extend(extension)

        self.atoms = atoms_list


class SurfaceRasterMetrics(analysis.PredictionMetrics):
    """This class analyzes the surface interaction of an additive with a surface.
    It is used to evaluate how well the surface structure is learned during training.
    Note that the bulk atoms should not be rattled in the SurfaceRasterScan node.

    Attributes
    ----------
    scan_node: SurfaceRasterScan()
       The node used for generating the structures

    """

    scan_node: SurfaceRasterScan = zntrack.deps()

    def get_plots(self, save=False):
        super().get_plots(save=True)
        self.plots_dir.mkdir(exist_ok=True)

        # get positions
        pos = []
        for atoms in self.data.atoms:
            pos.append(atoms.positions[-1])
        pos = np.array(pos)

        shape = [len(self.scan_node.z_dist_list)]
        shape.append(self.scan_node.n_conf_per_dist[0])
        shape.append(self.scan_node.n_conf_per_dist[1])

        x_pos = np.reshape(pos[:, 0], shape)
        x_pos = x_pos[0]
        for j in range(x_pos.shape[1]):
            x_pos[j, :] = x_pos[j, 0]

        y_pos = np.reshape(pos[:, 1], shape)
        y_pos = y_pos[0]

        # get energy
        true_energies = np.reshape(self.energy_df["true"], shape)
        pred_energies = np.reshape(self.energy_df["prediction"], shape)

        # get forces
        shape.append(3)

        forces = []
        for true_data, pred_data in zip(self.x, self.y):
            forces.append([true_data.get_forces(), pred_data.get_forces()])

        forces = np.array(forces)[:, :, -1, :] * 1000
        true_forces = np.reshape(forces[:, 0, :], shape)
        pred_forces = np.reshape(forces[:, 1, :], shape)

        for i, distance in enumerate(self.scan_node.z_dist_list):
            plot_ture_vs_pred(
                x_pos,
                y_pos,
                [true_energies[i, :], pred_energies[i, :]],
                "energies",
                distance,
                plots_dir=self.plots_dir,
            )
            plot_ture_vs_pred(
                x_pos,
                y_pos,
                [true_forces[i, :, :, 2], pred_forces[i, :, :, 2]],
                "forces",
                distance,
                plots_dir=self.plots_dir,
            )


def plot_ture_vs_pred(x, y, z, name, height, plots_dir):
    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(8, 4))
    for i, ax in enumerate(axes.flat):
        cm = ax.pcolormesh(x, y, z[i])
        ax.axis("scaled")
        ax.set_xlabel(r"x position additiv $\AA$")
        ax.set_ylabel(r"y-position additiv $\AA$")
    axes[0].set_title(f"true-{name}")
    axes[1].set_title(f"predicted-{name}")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.015, 0.03, 0.87])
    fig.colorbar(cm, cax=cbar_ax)

    if name == "energies":
        cbar_ax.set_ylabel(r"Energy $E$ / meV/atom")
    if name == "forces":
        cbar_ax.set_ylabel(r"Magnetude of force per atom $|F|$ meV$ \cdot \AA^{-1}$")

    fig.suptitle(rf"Additive {height} $\AA$ over the surface")
    fig.savefig(plots_dir / f"{name}-{height}-heat.png")



class SurfaceAdditive(base.ProcessSingleAtom):
    """This class generates periodic structures by creating a vacuum slab in the
    z-direction and adding additives at various positions. It is useful for generating
    input structures for surface training simulations or in combination with the
    SurfaceRasterMetrics class to analyze how well surface interactions are captured
    in the training.

    Attributes
    ----------
    symbol: str
        ASE symbol representing the additives.
    z_dist_list: list[float]
         A list of z-distances at which additives will be added.
    n_conf_per_dist: list[int]
        The number of configurations to generate per z-distance.
    cell_fraction: list[float]
        Fractional scaling of the unit cell in x and y directions.
    random: bool
       If True, additives are placed randomly within the specified cell_fraction.
    max_rattel_shift: float
        Maximum random displacement for each atom.
    seed: int
        Seed for randomly distributing the additive.
    """

    symbol: str = zntrack.params()
    z_dist_list: list[float] = zntrack.params()
    min_dist: float = zntrack.params()
    n_conf_per_dist: int = zntrack.params(5)
    seed: int = zntrack.params(1)

    def run(self) -> None:
        rng = default_rng(self.seed)
        
        atoms = self.get_data()

        cell = atoms.cell
        cellpar = cell.cellpar()
        cell = np.array(cell)

        z_max = max(atoms.get_positions()[:, 2])
        atoms_list = []
        for z_dist in self.z_dist_list:
            if cellpar[2] < z_max + z_dist + 10:
                cellpar[2] = z_max + z_dist + 10
                new_cell = Cell.fromcellpar(cellpar)
                atoms.set_cell(new_cell)
                log.warning("vacuum was extended")

            current_num_strucs = 0
            scaled_positions = []
            
            diff = [self.min_dist / np.linalg.norm(cell[0, :2])/2, self.min_dist / np.linalg.norm(cell[1, :2])/2]
            
            while current_num_strucs < self.n_conf_per_dist:
                a_scaling = np.array(rng.uniform(diff[0], 1-diff[0]))
                b_scaling = np.array(rng.uniform(diff[1], 1-diff[1]))
                
                scaled_a_vec = a_scaling * cell[0, :2]
                scaled_b_vec = b_scaling * cell[1, :2]

                cart_pos = scaled_a_vec + scaled_b_vec

                if current_num_strucs == 0:
                    scaled_positions.append(cart_pos)
                    num_failed_iterations = 0
                    current_num_strucs += 1
                else:
                    dist = [abs(np.linalg.norm(cart_pos - x)) for x in scaled_positions]
                    
                    if np.any(np.array(dist) < self.min_dist):
                        num_failed_iterations +=1
                    else:
                        scaled_positions.append(cart_pos)
                        num_failed_iterations = 0
                        current_num_strucs += 1
                        
                if current_num_strucs == self.n_conf_per_dist:
                    break
                elif num_failed_iterations > 2000 :
                    raise ValueError(f'Cannot fit additives with distance {self.min_dist} into the cell; maximum is {current_num_strucs}')
            
            scaled_positions = np.array(scaled_positions)
            z_pos = np.full_like(scaled_positions[:, 0], z_max + z_dist)
            z_pos = z_pos.reshape(-1, 1)
            scaled_positions = np.concatenate((scaled_positions, z_pos), axis=1)
            
            atoms_list.append(atoms.copy())
            extension = ase.Atoms(
                f'{self.symbol}{len(scaled_positions)}', scaled_positions
            )
            atoms_list[-1].extend(extension)

        self.atoms = atoms_list
        

def y_rot_mat(angle_radians):
    rotation_matrix = np.array([[np.cos(angle_radians), 0, -np.sin(angle_radians)],
                                [0, 1, 0],
                                [np.sin(angle_radians), 0, np.cos(angle_radians)],
                                ])
    return rotation_matrix

def z_rot_mat(angle_radians):
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                                [np.sin(angle_radians), np.cos(angle_radians), 0],
                                [0, 0, 1],
                                ])
    return rotation_matrix

def position_velocitie_rotation(pos, velo, angle_degrees, rot_axis):
    rot_matrix = {"y": y_rot_mat,
                     "z": z_rot_mat}
    
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)
    
    rotation_matrix = rot_matrix[rot_axis](angle_radians)
    
    # Apply the rotation matrix to the vector
    rotated_pos = np.dot(rotation_matrix, pos)
    rotate_velo = np.dot(rotation_matrix, velo)
    return rotated_pos, rotate_velo


class PosVeloRotation(base.ProcessSingleAtom):
    """This class generates 
    """

    symbol: str = zntrack.params()
    y_rotation_angles: list[float] = zntrack.params()
    z_rotation_angles: list[float] = zntrack.params()
    position: list[float] = zntrack.params()           #np.array([0., 0., 8.0*Ang,])
    velocitie:  list[float] = zntrack.params()          #np.array([0., 0., -8000.0*m/s,])
    n_conf_per_dist: list[int] = zntrack.params([5, 5])
    cell_fraction: list[float] = zntrack.params([1, 1])

    output_file = zntrack.outs_path(zntrack.nwd / "structures.h5")

    def run(self) -> None:
        self.y_rotation_angles = np.array(self.y_rotation_angles)
        self.z_rotation_angles = np.array(self.z_rotation_angles)
        self.position = np.array(self.position)
        self.velocitie = np.array(self.velocitie)
        
        atoms = self.get_data()
        cell = atoms.cell
        cellpar = cell.cellpar()

        z_max = max(atoms.get_positions()[:, 2])
        if cellpar[2] < self.position[2] + 10:
            cellpar[2] = self.position[2] + 10
            new_cell = Cell.fromcellpar(cellpar)
            atoms.set_cell(new_cell)
            log.warning("vacuum was extended")

        cell = np.array(atoms.cell)  
        
        fraction = [(2*(self.n_conf_per_dist[0]))**-1, (2*(self.n_conf_per_dist[1]))**-1]
        a_scaling = np.linspace(fraction[0], 1-fraction[0], self.n_conf_per_dist[0])
        b_scaling = np.linspace(fraction[1], 1-fraction[1], self.n_conf_per_dist[1])

        a_vec = cell[0, :2] * self.cell_fraction[0]
        scaled_a_vecs = a_scaling[:, np.newaxis] * a_vec
        b_vec = cell[1, :2] * self.cell_fraction[1]
        scaled_b_vecs = b_scaling[:, np.newaxis] * b_vec

        self.atoms = []
        for a in scaled_a_vecs:
            for b in scaled_b_vecs:
                xy_impact_pos = np.array(a + b)

                for z_angle in self.z_rotation_angles:
                    for y_angle in self.y_rotation_angles:
                        self.atoms.append(atoms.copy())
                        
                        rot_pos, rot_velo = position_velocitie_rotation(self.position, self.velocitie, y_angle, "y")
                        rot_pos_z, rot_velo_z = position_velocitie_rotation(rot_pos, rot_velo, z_angle, "z")

                        final_pos = rot_pos_z + np.array([xy_impact_pos[0], xy_impact_pos[1], z_max])
                        additive = ase.Atoms(
                                self.symbol,
                                [final_pos],
                                velocities=rot_velo_z,
                        )
                        self.atoms[-1].extend(additive)

