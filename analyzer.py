import numpy as np
import pandas as pd
import shutil
from rdkit import Chem
from rdkit.Chem import AllChem
from numpy import linspace
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
import os
import subprocess
import scipy.spatial.transform as sst
import ipywidgets
os.environ['OVITO_GUI_MODE'] = '1'
from ovito.io import *
from ovito.modifiers import *
from ovito.data import *
from ovito.pipeline import *
from ovito.vis import *
from tabulate import tabulate
from EDA_analyzer.util import *
from tqdm import tqdm
import re
import time
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} elapsed time: {elapsed_time} seconds")
        return result
    return wrapper
class EDA_analyzer():
    def __init__(self,molecule,probe='Li',boundary=10,grid_spacing=1,sieve=5,cavity_thres=3,isovalue=(0.05,0.1),grids_source='cubic'):
        self.molecule = molecule
        self.probe = probe
        self.molecule_coordinates = []
        self.boundary = boundary
        self.isovalue = isovalue
        self.grids_source = grids_source
        self.grid_spacing = grid_spacing
        self.sieve = sieve
        self.cavity_thres = cavity_thres
        self.origin = []
        self.molecule_extractor()
        self.gridpoints_generator(grids_source=self.grids_source,isoval=self.isovalue,gfn=1,chrg=0)
        self.gridpoints_filter()
        self.gridpoints_exporter(self.gridpoints_filtered)
        print(15 * '-' + str(len(self.gridpoints_coordinate)) + ' gridpoints were generated to be filtered' + 15 * '-')
        print(15*'-'+str(len(self.gridpoints_filtered))+' gridpoints were generated after filtration'+15*'-')
    @timer
    def molecule_extractor(self):
        f = open(self.molecule, 'r')
        lines = f.readlines()
        del lines[0:2]
        for line in lines:
            self.molecule_coordinates.append(line.split())
        self.molecule_coordinates = pd.DataFrame(self.molecule_coordinates)
        self.molecule_coordinates.columns=['atom_name','X','Y','Z']
        self.origin = [pd.to_numeric(self.molecule_coordinates.iloc[:,i]).mean() for i in [1,2,3]]
        coordinates = self.molecule_coordinates.iloc[:,1:4].to_numpy(dtype=float)
        dist=[distance.euclidean(self.origin,coordinate) for coordinate in coordinates]
        self.grid_length = np.array(dist).max()*1.414 + self.boundary
    @timer
    def gridpoints_generator(self,grids_source = 'cubic',isoval=(0.05,0.1),gfn=1,chrg=0):
        multiwfn_dir = subprocess.run(['which', 'Multiwfn'], stdout=subprocess.PIPE).stdout.decode().strip()
        def run_multiwfn(file, isoval=(0.05, 0.1)):
            print(15 * '-' + 'wavefunction file found' + 15 * '-')
            module_dir = os.path.dirname(__file__)
            file_path = os.path.join(module_dir, 'den.txt')
            with open(file_path, 'r') as f:
                lines = f.readlines()
            lines.insert(3, f"{self.origin[0]} {self.origin[1]} {self.origin[2]}\n")
            lines.insert(4,
                         f"{int(self.grid_length / self.grid_spacing) + 1} {int(self.grid_length / self.grid_spacing) + 1} {int(self.grid_length / self.grid_spacing) + 1}\n")
            lines.insert(5,
                         f"{self.grid_length * 1.889725988 / 2} {self.grid_length * 1.889725988 / 2} {self.grid_length * 1.889725988 / 2}\n")
            with open(f'{module_dir}/{self.molecule.split(".")[0]}_den.txt', 'w') as f:
                f.writelines(lines)
            process = subprocess.Popen(f'{multiwfn_dir} {file} < {module_dir}/{self.molecule.split(".")[0]}_den.txt', \
                                       shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            stdout, stderr = process.communicate()
            df = pd.DataFrame(np.loadtxt('output.txt'))
            df1 = df[df[3] > isoval[0]]
            self.gridpoints_coordinate_den = df1[df1[3] < isoval[1]]
            self.gridpoints_coordinate_den = self.gridpoints_coordinate_den.reset_index(drop=True)
            self.gridpoints_coordinate_den.insert(0, 'atom_name', self.probe)
            self.gridpoints_coordinate_den.columns = ['atom_name', 'X', 'Y', 'Z', 'ele_density']
            self.gridpoints_coordinate = self.gridpoints_coordinate_den.iloc[:, 0:4]
            self.gridpoints_coordinate.columns = ['atom_name', 'X', 'Y', 'Z']
            os.remove('output.txt')
            os.remove(f'{module_dir}/{self.molecule.split(".")[0]}_den.txt')

        if grids_source == 'cubic':
            length  = self.grid_length
            spacing = self.grid_spacing
            x = linspace(self.origin[0] - length/2, self.origin[0] + length/2, int(length/spacing) + 1)
            y = linspace(self.origin[1] - length/2, self.origin[1] + length/2, int(length/spacing) + 1)
            z = linspace(self.origin[2] - length/2, self.origin[2] + length/2, int(length/spacing) + 1)
            self.X, self.Y, self.Z = np.meshgrid(x, y, z)
            grid = np.stack((self.X, self.Y, self.Z), axis=-1)
            self.gridpoints_coordinate = pd.DataFrame(np.reshape(grid, (-1, 3)))
            self.gridpoints_coordinate.insert(0,'atom_name',self.probe)
            self.gridpoints_coordinate.columns = ['atom_name', 'X', 'Y', 'Z']

        if grids_source == 'molden':
            run_multiwfn(f'{self.molecule.split(".")[0]}.molden',isoval=isoval)

        if grids_source == 'xtb':
            os.mkdir('xtb_temp')
            os.system(f'cp -f {self.molecule} xtb_temp')
            os.chdir('xtb_temp')
            xtb_dir = subprocess.run(['which', 'xtb'], stdout=subprocess.PIPE).stdout.decode().strip()
            subprocess.run([f'{xtb_dir}', self.molecule, '--molden', f'--gfn {gfn}',f'--chrg {chrg}'], stdout=subprocess.DEVNULL)
            os.system(f'cp -f molden.input ../{self.molecule.split(".")[0]}.molden')
            os.chdir('..')
            os.system('rm -rf xtb_temp')
            run_multiwfn(f'{self.molecule.split(".")[0]}.molden', isoval=isoval)

        if grids_source == 'turbomole':
            x2t_dir = subprocess.run(['which', 'x2t'], stdout=subprocess.PIPE).stdout.decode().strip()
            define_dir = subprocess.run(['which', 'define'], stdout=subprocess.PIPE).stdout.decode().strip()
            ridft_dir = subprocess.run(['which', 'ridft'], stdout=subprocess.PIPE).stdout.decode().strip()
            module_dir = os.path.dirname(__file__)
            molden_path = os.path.join(module_dir, 'molden.inp')
            def_path = os.path.join(module_dir, 'def')
            os.mkdir('turbomole_temp')
            os.system(f'cp -f {self.molecule} turbomole_temp')
            os.chdir('turbomole_temp')
            commands = [f'{x2t_dir} {self.molecule} > coord', f'{define_dir} < {def_path}',
                        f'{ridft_dir} > ridft.out', f'tm2molden < {molden_path}' \
                , f'mv molden.input ../{self.molecule.split(".")[0]}.molden']
            run_commands(commands)
            os.chdir('..')
            run_multiwfn(f'{self.molecule.split(".")[0]}.molden', isoval=isoval)
            os.system('rm -rf turbomole_temp')

    @timer
    def gridpoints_filter(self):
        gridpoints = self.gridpoints_coordinate.iloc[:,1:4].to_numpy(dtype=float)
        molecule   = self.molecule_coordinates.iloc[:,1:4].to_numpy(dtype=float)
        self.dist = distance.cdist(gridpoints, molecule, metric='euclidean')
        gridpoints_coordinate = self.gridpoints_coordinate.iloc[:,1:4].to_numpy()
        filtered_index = np.where(np.all((self.dist >= self.cavity_thres) , axis=1))[0]
        cavity_index = np.where(np.any((self.dist < self.cavity_thres) , axis=1))[0]
        self.gridpoints_cavity = pd.DataFrame(gridpoints_coordinate[cavity_index])
        self.gridpoints_cavity.insert(0, 'atom_name', self.probe)
        self.gridpoints_cavity.columns = ['atom_name', 'X', 'Y', 'Z']
        self.gridpoints_filtered_1 = gridpoints_coordinate[filtered_index]
        self.dist_filtered_1 = distance.cdist(self.gridpoints_filtered_1, molecule, metric='euclidean')
        dist_filtered_min_1 = np.amin(self.dist_filtered_1, axis=1)
        filtered_index = np.where(dist_filtered_min_1 < self.sieve)[0]
        self.gridpoints_filtered = pd.DataFrame(self.gridpoints_filtered_1[filtered_index])
        self.dist_filtered = distance.cdist(self.gridpoints_filtered, molecule, metric='euclidean')
        self.gridpoints_filtered.insert(0, 'atom_name', self.probe)
        self.gridpoints_filtered.columns = ['atom_name', 'X', 'Y', 'Z']
    
    def gridpoints_exporter(self,gridpoints,name='grids'):
        with open(f'{self.molecule.split(".")[0]}_{name}.xyz','w') as f:
            f.write(str(len(gridpoints)))
            f.write('\n\n')
            np.savetxt(f,gridpoints.to_numpy(),fmt='%s')
    def gridpoints_visualizer(self,axis,animation_speed,frame,fps,viewpoint,ovito=[False,False,(600,500),False],*val,eda_val='Eint_total,gas'):
        molecule = self.molecule_coordinates
        gridpoints = self.gridpoints_filtered
        molecule_atom_name = molecule.iloc[:,0:1]
        molecule_reset = molecule.iloc[:,1:4].to_numpy(dtype=float)-self.origin
        gridpoints_atom_name = gridpoints.iloc[:,0:1]
        gridpoints_reset = gridpoints.iloc[:,1:4].to_numpy(dtype=float)-self.origin
        def rotation(reset,atom_name,axis,name,animation_speed,*val):
            if name + '_rotations.xyz' in list(os.walk('./'))[0][2]:
                os.remove(name + '_rotations.xyz')
            angle = 0
            rotated_coordinates = []
            while angle <= 360:
                rotate_angle = np.array([0,0,0])
                rotate_angle[axis] =angle
                rotation_matrix = sst.Rotation.from_rotvec(np.array([0, 0, angle])).as_matrix()
                rotated = np.dot(reset,rotation_matrix)
                if len(val) != 0:
                    rotated = np.column_stack((rotated,self.gridpoints_normalized[list(val)].to_numpy()))
                    rotated_coordinates.append(rotated)
                else:
                    rotated_coordinates.append(rotated)
                angle += animation_speed
            for rotated_coordinate in rotated_coordinates:
                xyz_file = np.column_stack((atom_name,rotated_coordinate))
                with open(name+'_rotations.xyz', "a") as f:
                    f.write(str(len(rotated_coordinate)))
                    f.write('\n\n')
                    np.savetxt(f,xyz_file,fmt='%s')
        if ovito[0] == False:
            rotation(molecule_reset,molecule_atom_name,axis,self.molecule.split('.')[0],animation_speed)
            rotation(gridpoints_reset,gridpoints_atom_name,axis,self.molecule.split('.')[0]+'_gridpoints',animation_speed,*val)
        else:
            molecule_v = import_file(self.molecule.split('.')[0]+'_rotations.xyz', columns = ['Particle Type', 'Position.X', 'Position.Y', 'Position.Z'])
            def modify_pipeline_input(frame: int, data: DataCollection):
                module_dir = os.path.dirname(__file__)
                file_path = os.path.join(module_dir, 'elements.csv')
                atom_type = pd.read_csv(file_path)
                total_atoms = np.unique(self.molecule_coordinates['atom_name'].to_numpy())
                for atom in total_atoms:
                    data.particles_.particle_types_.type_by_name_(atom).radius = atom_type.loc[atom_type['atom_name'] == atom]['size'].to_numpy()[0]
                    data.particles_.particle_types_.type_by_name_(atom).color = tuple([atom_type.loc[atom_type['atom_name'] == atom][i].to_numpy()[0] for i in ['R','G','B']])
            molecule_v.modifiers.append(modify_pipeline_input)
            data = molecule_v.compute() # Evaluate new pipeline to gain access to visual elements associated with the imported data objects.
            data.particles.vis.radius = 0.2
            data.cell.vis.enabled = False
            del data # Done accessing input DataCollection of pipeline.
            mod = CreateBondsModifier()
            mod.mode = CreateBondsModifier.Mode.VdWRadius
            mod.vis.width = 0.2
            molecule_v.modifiers.append(mod)
            molecule_v.add_to_scene()
            vp = viewpoint
            if ovito[1] == True:
                if ovito[3] == True:
                    vp.render_anim(size=ovito[2], filename=self.molecule.split('.')[0] + '_rotations.gif', fps=fps)
                else:
                    mol = self.molecule.split('.')[0]
                    vp.render_image(size=ovito[2], frame=frame, filename=f'{mol}_{frame}.png')
            gridpoints_v = import_file(self.molecule.split('.')[0]+'_gridpoints_rotations.xyz', columns = ['Particle Type', 'Position.X', 'Position.Y', 'Position.Z']+list(val))
            def modify_pipeline_input(frame: int, data: DataCollection):
                if len(val) == 0:
                    data.particles_.particle_types_.type_by_name_(self.probe).radius = 0.12
                else:
                    data.particles_.particle_types_.type_by_name_(self.probe).radius = 1.9
                    data.particles_.particle_types_.type_by_name_(self.probe).shape = ParticlesVis.Shape.Sphere
            gridpoints_v.modifiers.append(modify_pipeline_input)
            data_1 = gridpoints_v.compute() # Evaluate new pipeline to gain access to visual elements associated with the imported data objects.
            if len(val) == 0:
                data_1.particles.vis.radius = 0.2
            else:
                data_1.particles.vis.enabled = False
            data_1.cell.vis.enabled = False
            del data_1 # Done accessing input DataCollection of pipeline.
            if len(val) != 0:
                top = self.gridpoints_normalized[eda_val].median() + self.gridpoints_normalized[eda_val].std()
                btm = self.gridpoints_normalized[eda_val].median() - self.gridpoints_normalized[eda_val].std()
                gridpoints_v.modifiers.append(ColorCodingModifier(
                    property = eda_val,
                    start_value = max(0,btm),
                    end_value =  min(1,top),
                    gradient = ColorCodingModifier.Rainbow()))
                mod2 = ConstructSurfaceModifier()
                mod2.method = ConstructSurfaceModifier.Method.GaussianDensity
                mod2.transfer_properties = True
                mod2.grid_resolution = 100
                mod2.radius_scaling = 0.3
                mod2.isolevel = 6.0
                mod2.vis.show_cap = False
                mod2.vis.surface_transparency = 0.5
                gridpoints_v.modifiers.append(mod2)
            else:
                pass
            gridpoints_v.add_to_scene()
            if ovito[3] == True:
                vp.render_anim(size=ovito[2], filename=self.molecule.split('.')[0] + '_'+eda_val+'_gridpoints_rotations.gif', fps=fps)
            else:
                mol = self.molecule.split('.')[0]
                # vp.render_image(size=ovito[2], frame=frame, filename=f'{mol}_{frame}_{eda_val}_gridpoints.png')
                widget = vp.create_jupyter_widget(layout = ipywidgets.Layout(width='2000px', height='500px'))
                display(widget)
            molecule_v.remove_from_scene()
            gridpoints_v.remove_from_scene()
    @timer
    def run_LJ(self):
        grid_coord = np.loadtxt(self.molecule.split('.')[0]+'_grids.xyz',skiprows=2,usecols=[1,2,3])
        mol_coord = np.loadtxt(self.molecule,skiprows=2,usecols=[1,2,3])
        atom_grid = np.loadtxt(self.molecule.split('.')[0]+'_grids.xyz',dtype='str',skiprows=2,usecols=[0])
        atom_coord = np.loadtxt(self.molecule,dtype='str',skiprows=2,usecols=[0])
        dist_arr = self.dist_filtered.T
        module_dir = os.path.dirname(__file__)
        file_path = os.path.join(module_dir, 'LJ_potentials.csv')  
        LJ_potentials = pd.read_csv(file_path)
        map_dict = LJ_potentials.set_index('Atom').to_dict()['Epsilon']
        epsilon_mol = np.array([map_dict.get(i,i) for i in atom_coord])
        epsilon_grid = np.array([map_dict.get(i,i) for i in atom_grid])
        sigma_mol = np.array([map_dict.get(i,i) for i in atom_coord])
        sigma_grid = np.array([map_dict.get(i,i) for i in atom_grid])
        ep_mix = np.sqrt(np.einsum('i,j-> ij', epsilon_mol,epsilon_grid))
        sig_mix = (np.add.outer(sigma_mol,sigma_grid))/2
        self.gridpoints_LJ = 4*ep_mix*(np.power((sig_mix/dist_arr),6) - np.power((sig_mix/dist_arr),12))
        self.gridpoints_LJ = pd.DataFrame(np.einsum('ij->j',self.gridpoints_LJ))
        self.gridpoints_LJ.columns = ['LJ_energy']
        scaler = MinMaxScaler()
        self.gridpoints_normalized_LJ =pd.DataFrame(scaler.fit_transform(self.gridpoints_LJ['LJ_energy'].to_numpy().reshape(-1,1)))
        self.gridpoints_normalized_LJ.columns = ['LJ_energy']
        self.gridpoints_normalized = self.gridpoints_normalized_LJ
    @timer
    def run_xTB(self,gfn=1,chrg=1):
        xtb_dir = subprocess.run(['which','xtb'],stdout=subprocess.PIPE).stdout.decode().strip()
        xtbiff_dir = subprocess.run(['which','xtbiff'],stdout=subprocess.PIPE).stdout.decode().strip()
        working_directory, path_1, path_2, path_3 = get_paths()
        print('----------------------GRID EDA INITIATED-------------------')
        os.system('mkdir 1')
        os.system('cp '+self.molecule+ ' 1')
        os.chdir(path_1)
        subprocess.run([xtb_dir, self.molecule, '--lmo',f'--gfn {gfn}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        os.rename('xtblmoinfo','1')
        os.chdir(working_directory)
        os.system('mkdir 2 3')
        os.system('mv -f ' + path_1 + '/1 ' + path_3)
        self.gridpoints_xTB = []
        with tqdm(total=(len(self.gridpoints_filtered)), desc='Grid Initiated', unit="gridpoint") as pbar:
            for self.i,gridpoint in enumerate(self.gridpoints_filtered.to_numpy()):
                os.chdir(path_2)
                with open('gridpoint.xyz','w') as f:
                    f.write('1')
                    f.write('\n\n')
                    np.savetxt(f,self.gridpoints_filtered.iloc[self.i:self.i+1].to_numpy(),fmt='%s')
                subprocess.run([xtb_dir, 'gridpoint.xyz',f'--chrg {chrg}' ,'--lmo',f'--gfn {gfn}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                os.rename('xtblmoinfo','2')
                os.system('mv -f '+'2 '+ path_3)
                os.chdir(path_3)
                with open('int.txt','w') as f:
                    subprocess.run([xtbiff_dir, '1','2' ,'-sp'],stdout=f)
                with open('int.txt', 'r') as f:
                    lines = f.readlines()
                    gridpoint_EDA =np.array([line.split(':')[1] for line in lines[-11:-1]]).astype(float)
                self.gridpoints_xTB.append(gridpoint_EDA)
                pbar.update(1)
            with open('int.txt', 'r') as f:
                lines = f.readlines()
                self.columns =np.array(["_".join((line.split(':')[0]).split()) for line in lines[-11:-1]]).tolist()
            self.gridpoints_xTB = pd.DataFrame(self.gridpoints_xTB)
            self.gridpoints_xTB.columns = self.columns
            os.chdir(working_directory)
            os.system('rm -rf 1 2 3')
            scaler = MinMaxScaler()
            self.gridpoints_normalized_xTB =pd.DataFrame(scaler.fit_transform(self.gridpoints_xTB[self.columns]))
            self.gridpoints_normalized_xTB.columns = self.columns
            self.gridpoints_normalized = self.gridpoints_normalized_xTB
            with open(f"{self.molecule.split('.')[0]}_xTB_energy_values.md", "w") as f:
                f.write(tabulate(self.gridpoints_xTB, headers=self.columns, tablefmt="simple", floatfmt=".6f", showindex=False,colalign=("center",) * len(self.columns)))
            df2 = pd.concat([self.gridpoints_filtered, self.gridpoints_xTB], axis=1)
            with open(f'{self.molecule.split(".")[0]}_xTB.xyz', 'w') as f:
                f.write(str(len(df2)))
                f.write('\n\n')
                np.savetxt(f, df2.to_numpy(), fmt='%s')
            print('----------------------GRID EDA COMPLETED-------------------')
    @timer
    def run_Turbomole(self,cores=8):
        working_directory, path_1, path_2, path_3 = get_paths()
        if 'def1' or 'def2' or 'def3' not in list(os.walk('./'))[0][2]:
            module_dir = os.path.dirname(__file__)
            def1 = os.path.join(module_dir, 'def1')
            def2 = os.path.join(module_dir, 'def2')
            def3 = os.path.join(module_dir, 'def3')
        else:
            def1 = '../def1'
            def2 = '../def2'
            def3 = '../def3'
        file_path = os.path.join(module_dir, 'def1')
        frag_a =[
                f"export PARNODES={cores}",
                f"mkdir 1", # create directory 1
                f"cp -f {self.molecule} {path_1}", # copy the molecule file to path_1
                f"cd {path_1}", # change the working directory to path_1
                f"x2t {self.molecule} > coord", # convert xyz coordinates to internal coordinates
                f"define < {def1} > out 2> out1", # run the define command with def1 file as input
                "head -n -1 control > temp.txt",
                "echo '$scfdenapproxl 0' >> temp.txt",
                "echo '$end' >> temp.txt",
                "rm control",# remove the control file
                "mv temp.txt control",
                "ridft > ridft.out 2> out1", # run the ridft command
                f"cd {working_directory}",# change the working directory to the working directory
                "mkdir 2 3", # create directory 2 and 3
                f"cp -f {def2} {path_2}"]
        frag_b =[
                f"export PARNODES={cores}",
                "x2t coord.xyz > coord",  # convert xyz coordinates to internal coordinates
                f"define < {def2} > out 2> out1",  # run the define command with def1 file as input
                "head -n -1 control > temp.txt",
                "echo '$scfdenapproxl 0' >> temp.txt",
                "echo '$end' >> temp.txt",
                "rm control",
                "mv temp.txt control",
                "ridft > ridft.out 2> out1",  # run the ridft command
                f"cd {working_directory}"]
        Supramolecule = [
                f"export PARNODES={cores}",
                "x2t coord.xyz > coord",  # convert xyz coordinates to Turbomol coordinates
                f"define < {def3} > out 2> out1",  # run the define command with def1 file as input
                "head -n -1 control > temp.txt",  # remove the last line of the control file
                "echo '$scfdenapproxl 0' >> temp.txt",  # add the scfdenapproxl line to the temp file
                "echo '$subsystems' >> temp.txt",  # add the subsystems line to the temp file
                f"echo ' molecule#1 file={path_1}/control' >> temp.txt",  # add the molecule#1 line to the temp file
                f"echo ' molecule#2 file={path_2}/control' >> temp.txt",  # add the molecule#2 line to the temp file
                "echo '$end' >> temp.txt",  #
                "rm control",  # remove the control file
                "mv temp.txt control",  # replace the control file with the new control file
                'promowa > out 2> out1',
                "ridft > ridft.out 2> out"]
        run_commands(frag_a, output_file="output.txt")
        eda_energy_df = pd.DataFrame()
        print('----------------------GRID EDA INITIATED-------------------')
        with tqdm(total=(len(self.gridpoints_filtered)), desc='Grid Initiated', unit="gridpoint") as pbar:
            for i, gridpoint in enumerate(self.gridpoints_filtered.to_numpy()):
                pbar.set_description(f"grid {i + 1} of {len(self.gridpoints_filtered) + 1}")
                os.chdir(path_2)
                with open('coord.xyz', 'w') as f:
                    f.write('1')
                    f.write('\n\n')
                    np.savetxt(f, self.gridpoints_filtered.iloc[i:i + 1].to_numpy(), fmt='%s')
                run_commands(frag_b, output_file="output.txt")
                os.chdir(path_3)
                combined_coordinates = np.vstack((self.molecule_coordinates.to_numpy(), self.gridpoints_filtered.iloc[i:i + 1].to_numpy()))
                xyz_generator(f'{len(self.molecule_coordinates) + 1}', combined_coordinates)
                run_commands(Supramolecule, output_file="output.txt")
                df = extract_energy_values('ridft.out', 'dscf_problem')
                df["grid_index"] = i
                df.set_index("grid_index", inplace=True)
                eda_energy_df = pd.concat([eda_energy_df, df], ignore_index=False)
                os.chdir(str(working_directory))
                # print(f'        -------GRIDPOINT {i+1} of {len(test.gridpoints_filtered)+1} being Calculated-------')
                with open(f"energy_values_{self.molecule.split('.')[0]}.md", "w") as f:
                    f.write(tabulate(eda_energy_df,
                                     headers=["Grid_Index", "Tot", "Electro", "Nuc_Nuc", "1e", "2e", "Exc_Rep", "Exc",
                                              "Rep", "Orb_Relax", "Corr", "Disp"],
                                     tablefmt="simple", floatfmt=".10f",
                                     colalign=(
                                     "center", "center", "center", "center", "center", "center", "center", "center",
                                     "center", "center", "center", "center")))
                os.system('rm -r 2 3')
                os.system('mkdir 2 3')
                pbar.update(1)
        print('---------------------GRID EDA COMPLETED---------------------')


    def xyz_exporter(self,axis,animation_speed,*val,vp=Viewport(type = Viewport.Type.Front,fov = 11,camera_pos = (0,0,0),camera_dir = (1,0,0))):
        self.gridpoints_visualizer(axis, animation_speed,0, 5, vp, [False,False,(1, 2),False], *val)
    @timer
    def anime_visualizer(self,axis,angle,fps,figsize:tuple,*val,mol=False,label='Eint_total,gas',exporter=True,vp=Viewport(type = Viewport.Type.Front,fov = 11,camera_pos = (0,0,0),camera_dir = (1,0,0))):
        if exporter == True:
            self.xyz_exporter(axis,angle,*val)
        self.gridpoints_visualizer(0,3.6,0,fps,vp,[True,mol,figsize,True],*val,eda_val=label)
    @timer
    def image_visualizer(self,axis,angle,frame,figsize:tuple,*val,mol=True,label='Eint_total,gas',vp=Viewport(type = Viewport.Type.Front,fov = 11,camera_pos = (0,0,0),camera_dir = (1,0,0))):
        self.xyz_exporter(axis, angle, *val)
        self.gridpoints_visualizer(0,3.6,frame,1,vp,[True,mol,figsize,False],*val,eda_val=label)



class SmilesToXYZ():
    def __init__(self,smiles,file,conformer_search=False):
        self.smiles = smiles
        self.xtb_dir = subprocess.run(['which','xtb'],stdout=subprocess.PIPE).stdout.decode().strip()
        self.file = file
        self.__call__()
        self.crest_dir = subprocess.run(['which', 'crest'], stdout=subprocess.PIPE).stdout.decode().strip()
        if conformer_search == True:
            self.conformer_search()
            self.conformer_sorting()
    def __call__(self,gfn=2):
        m = Chem.MolFromSmiles(self.smiles)
        m2=Chem.AddHs(m)
        AllChem.EmbedMolecule(m2)
        with open(self.file, 'w') as f:
            f.write(Chem.MolToXYZBlock(m2, confId=-1))
        path = str(os.getcwd())
        path_opt = path+'/opt'
        os.system('mkdir opt')
        os.system('mv '+self.file+ ' opt')
        os.chdir(path_opt)
        subprocess.run([self.xtb_dir, self.file, '--opt',f'--gfn {gfn}'],stdout=subprocess.DEVNULL)
        os.system(f'mv xtbopt.xyz {path}')
        os.chdir(path)
        os.rename('xtbopt.xyz',self.file)
        os.system('rm -rf opt')
    @timer
    def conformer_search(self,gfn=['1','2'],thread=8,sol='h2o'):
        os.system(f'rm -rf {self.file.split(".")[0]}_conformers')
        print('Conformer search is started')
        os.mkdir(f'{self.file.split(".")[0]}_conformers')
        os.system(f'cp {self.file} md.inp {self.file.split(".")[0]}_conformers')
        os.chdir(f'{self.file.split(".")[0]}_conformers')
        os.system(f'{self.xtb_dir} {self.file} --input md.inp --omd --gfn {gfn[0]} --alpb {sol} --T {thread} > md.out')
        os.rename('xtb.trj','traj.xyz')
        subprocess.run([self.crest_dir,'--mdopt','traj.xyz',f'--gfn {gfn[1]}',f'--alpb {sol}',f'--T {thread}','--niceprint'],stdout= subprocess.DEVNULL)
        os.rename('crest_ensemble.xyz',f'{self.file.split(".")[0]}_ensemble.xyz')
    def conformer_sorting(self):
        with open(f'{self.file.split(".")[0]}_sorting.out', 'w') as f:
            subprocess.run([self.crest_dir, '--cregen', f'{self.file.split(".")[0]}_ensemble.xyz'], stdout=f)
            os.rename('crest.energies', f'{self.file.split(".")[0]}_energy.txt')
            os.rename('crest_ensemble.xyz',f'{self.file.split(".")[0]}_ensemble.sorted.xyz')
            os.rename('crest_best.xyz',f'{self.file.split(".")[0]}_best.xyz')
            os.system(f'mv {self.file.split(".")[0]}_ensemble.sorted.xyz {self.file.split(".")[0]}_energy.txt {self.file.split(".")[0]}_best.xyz  ../')
            os.chdir('../')
            # os.system(f'rm -rf {self.file.split(".")[0]}_conformers')







