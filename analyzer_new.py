import numpy as np
import pandas as pd
import shutil
from tabulate import tabulate
import select
import glob
import re
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
    def __init__(self,molecule,probe='Li',boundary=10,grid_spacing=1,sieve=5,cavity_thres=3):
        self.molecule = molecule
        self.probe = probe
        self.molecule_coordinates = []
        self.boundary = boundary
        self.grid_spacing = grid_spacing
        self.sieve = sieve
        self.cavity_thres = cavity_thres
        self.origin = []
        self.molecule_extractor()
        self.gridpoints_generator()
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
    def gridpoints_generator(self):
        length  = self.grid_length
        spacing = self.grid_spacing
        x = linspace(self.origin[0] - length/2, self.origin[0] + length/2, int(length/spacing) + 1)
        y = linspace(self.origin[1] - length/2, self.origin[1] + length/2, int(length/spacing) + 1)
        z = linspace(self.origin[2] - length/2, self.origin[2] + length/2, int(length/spacing) + 1)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z)
        grid = np.stack((self.X, self.Y, self.Z), axis=-1)
        self.gridpoints_coordinate = pd.DataFrame(np.reshape(grid, (-1, 3)))
        self.gridpoints_coordinate.insert(0,'atom_name',self.probe)
    @timer
    def gridpoints_filter(self):
        gridpoints = self.gridpoints_coordinate.iloc[:,1:4].to_numpy(dtype=float)
        molecule   = self.molecule_coordinates.iloc[:,1:4].to_numpy(dtype=float)
        self.dist = distance.cdist(gridpoints, molecule, metric='euclidean')
        gridpoints_coordinate = self.gridpoints_coordinate.iloc[:,1:4].to_numpy()
        filtered_index = np.where(np.all((self.dist >= self.cavity_thres) , axis=1))[0]
        self.gridpoints_filtered_1 = gridpoints_coordinate[filtered_index]
        self.dist_filtered_1 = distance.cdist(self.gridpoints_filtered_1, molecule, metric='euclidean')
        dist_filtered_min_1 = np.amin(self.dist_filtered_1, axis=1)
        filtered_index = np.where(dist_filtered_min_1 < self.sieve)[0]
        self.gridpoints_filtered = pd.DataFrame(self.gridpoints_filtered_1[filtered_index])
        self.dist_filtered = distance.cdist(self.gridpoints_filtered, molecule, metric='euclidean')
        self.gridpoints_filtered.insert(0, 'atom_name', self.probe)
        self.gridpoints_filtered.columns = ['atom_name', 'X', 'Y', 'Z']

    
    def gridpoints_exporter(self,gridpoints):
        gridpoints.columns = ['atom_name','X','Y','Z']
        np.savetxt(self.molecule.split('.')[0]+'_grids.xyz',gridpoints.to_numpy(),fmt='%s')
        with open(self.molecule.split('.')[0]+'_grids.xyz', 'r') as file:
            contents = file.read()
        contents = str(len(gridpoints))+'\n\n' + contents
        with open(self.molecule.split('.')[0]+'_grids.xyz', 'w') as file:
            file.write(contents)
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
                    data.particles_.particle_types_.type_by_name_(self.probe).radius = 0.02
            gridpoints_v.modifiers.append(modify_pipeline_input)
            data_1 = gridpoints_v.compute() # Evaluate new pipeline to gain access to visual elements associated with the imported data objects.
            data_1.particles.vis.radius = 0.2
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
                mod2.radius = 5
                mod2.smoothing_level = 35
                mod2.transfer_properties = True
                mod2.vis.show_cap = False
                mod2.vis.surface_transparency = 0.6
                gridpoints_v.modifiers.append(mod2)
            else:
                pass
            gridpoints_v.add_to_scene()
            if ovito[3] == True:
                vp.render_anim(size=ovito[2], filename=self.molecule.split('.')[0] + '_'+eda_val+'_gridpoints_rotations.gif', fps=fps)
            else:
                mol = self.molecule.split('.')[0]
                vp.render_image(size=ovito[2], frame=frame, filename=f'{mol}_{frame}_gridpoints.png')
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

    def get_paths(self,*args):
        working_directory = os.getcwd()
        for arg in args:
            locals()[f'path_{arg}'] = os.path.join(working_directory, arg)
            return locals()[f'path_{arg}']
        return working_directory

    def run_Jobs(self,frag_A,frag_B,Supramolecule,output_name,output_types,headers):

        def file_to_command(command_list, file, group_name):
            with open(file, "r") as f:
                command_list[group_name].extend(f.readlines())
                return command_list

        def run_commands(commands, timeout=0, log=False):
            output_list = []
            process = subprocess.Popen(["bash"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       universal_newlines=True)
            for i, command in enumerate(commands):
                process.stdin.write(f"{command}\n")
                process.stdin.flush()
                if log == True:
                    rlist, _, _ = select.select([process.stdout, process.stderr], [], [], timeout)
                    for r in rlist:
                        if r is process.stdout:
                            print(r)
                            output = process.stdout.readline()
                            output_list.append(["Command " + str(i + 1), command, "stdout", output])
                        else:
                            error = process.stderr.readline()
                            output_list.append(["Command " + str(i + 1), command, "stderr", error])
            stdout, stderr = process.communicate()
            headers = ["Command Number", "Command", "Output Type", "Output"]
            if log == True:
                with open('output.txt', "w") as f:
                    f.write(tabulate(output_list, headers, tablefmt="fancy_grid"))

        #output_tpyes will be a list containing any thing you want to extract from the output file
        def output_parser(file_name: str, error_file: str, output_types: list) -> pd.DataFrame:
            output_values = {output_type: 0 for output_type in output_types}
            if os.path.isfile(error_file):
                df = pd.DataFrame(output_values, index=["0"])
                return df
            else:
                with open(file_name, 'r') as f:
                    lines = f.read()
                    output_types_re = "|".join(output_types)
                    matches = re.finditer(output_types_re, lines)
                    for match in matches:
                        output_type = match.group()
                        match = re.search(r'-?\d+\.\d+', lines[match.start():])
                        if match:
                            output_values[output_type] = match.group()

                df = pd.DataFrame(output_values, index=["0"])
                return df

        working_directory, path_1, path_2, path_3 = get_paths()
        run_commands(frag_A)
        eda_energy_df = pd.DataFrame()
        print('---------------------EDA INITIATED---------------------')
        for i, gridpoint in enumerate(self.gridpoints_filtered.to_numpy()):
            os.chdir(path_2)
            with open('coord.xyz', 'w') as f:
                f.write('1')
                f.write('\n\n')
                np.savetxt(f, self.gridpoints_filtered.iloc[i:i + 1].to_numpy(), fmt='%s')
            run_commands(frag_B)
            os.chdir(path_3)
            with open('coord.xyz', 'w') as f:
                f.write(f'{len(self.molecule_coordinates)+1}')
                f.write('\n\n')
                combined_coordinates = np.concatenate((self.molecule_coordinates, self.gridpoints_filtered.iloc[i:i + 1].to_numpy()), axis=0)
                np.savetxt(f, combined_coordinates, fmt='%s')
            run_commands(Supramolecule)
            df = output_parser(output_name, 'dscf_problem',output_types)
            df["grid_index"] = i
            df.set_index("grid_index", inplace=True)
            eda_energy_df = pd.concat([eda_energy_df, df], ignore_index=False)
            os.chdir(str(working_directory))
            with open("energy_values.md", "w") as f:
                f.write(tabulate(eda_energy_df,
                                 headers=headers,
                                 tablefmt="simple", floatfmt=".6f",
                                 colalign=(
                                 "center", "center", "center", "center", "center", "center", "center", "center",
                                 "center", "center", "center", "center")))
            os.system('rm -r 2 3')
            os.system('mkdir 2 3')
        print('---------------------EDA COMPLETED---------------------')




    def run_Turbomole(self):

          
         
                
                    















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
    def __init__(self,smiles,file,conformer_search=False,sol='h20'):
        self.smiles = smiles
        self.xtb_dir = subprocess.run(['which','xtb'],stdout=subprocess.PIPE).stdout.decode().strip()
        self.file = file
        self.sol = sol
        self.__call__()
        self.crest_dir = subprocess.run(['which', 'crest'], stdout=subprocess.PIPE).stdout.decode().strip()
        if conformer_search == True:
            self.conformer_search(self.sol)
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






