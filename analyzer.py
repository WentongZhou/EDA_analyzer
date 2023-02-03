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
import multiprocessing
import concurrent.futures
import scipy.spatial.transform as sst
import ipywidgets
os.environ['OVITO_GUI_MODE'] = '1'
from ovito.io import *
from ovito.modifiers import *
from ovito.data import *
from ovito.pipeline import *
from ovito.vis import *
import time
np.set_printoptions(precision=6)
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
        self.gridpoints_coordinate.columns = ['atom_name', 'X', 'Y', 'Z']
        self.gridpoints_coordinate.sort_values(by=['X', 'Y', 'Z'], ascending=[True, True, True], inplace=True, ignore_index=True)
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
    @timer
    def run_xTB(self,gfn=1,chrg=1):
        xtb_dir = subprocess.run(['which','xtb'],stdout=subprocess.PIPE).stdout.decode().strip()
        xtbiff_dir = subprocess.run(['which','xtbiff'],stdout=subprocess.PIPE).stdout.decode().strip()
        path = str(os.getcwd())
        path_1 = path + '/1'
        path_2 = path + '/2'
        path_3 = path + '/3'
        os.system('mkdir 1')
        os.system('cp '+self.molecule+ ' 1')
        os.chdir(path_1)
        subprocess.run([xtb_dir, self.molecule, '--lmo',f'--gfn {gfn}'],stdout=subprocess.DEVNULL)
        os.rename('xtblmoinfo','1')
        os.chdir(path)
        os.system('mkdir 2 3')
        os.system('mv -f ' + path_1 + '/1 ' + path_3)
        self.gridpoints_xTB = []
        for self.i,gridpoint in enumerate(self.gridpoints_filtered.to_numpy()):
            os.chdir(path_2)
            with open('gridpoint.xyz','w') as f:
                f.write('1')
                f.write('\n\n')
                np.savetxt(f,self.gridpoints_filtered.iloc[self.i:self.i+1].to_numpy(),fmt='%s')
            subprocess.run([xtb_dir, 'gridpoint.xyz',f'--chrg {chrg}' ,'--lmo',f'--gfn {gfn}'],stdout=subprocess.DEVNULL)
            os.rename('xtblmoinfo','2')
            os.system('mv -f '+'2 '+ path_3)
            os.chdir(path_3)
            with open('int.txt','w') as f:
                subprocess.run([xtbiff_dir, '1','2' ,'-sp'],stdout=f)
            with open('int.txt', 'r') as f:
                lines = f.readlines()
                gridpoint_EDA =np.array([line.split(':')[1] for line in lines[-11:-1]]).astype(float)
            self.gridpoints_xTB.append(gridpoint_EDA)
        with open('int.txt', 'r') as f:
            lines = f.readlines()
            self.columns =np.array(["_".join((line.split(':')[0]).split()) for line in lines[-11:-1]]).tolist()
        self.gridpoints_xTB = pd.DataFrame(self.gridpoints_xTB)
        self.gridpoints_xTB.columns = self.columns
        os.chdir(path)
        os.system('rm -rf 1 2 3')
        scaler = MinMaxScaler()
        self.gridpoints_normalized_xTB =pd.DataFrame(scaler.fit_transform(self.gridpoints_xTB[self.columns]))
        self.gridpoints_normalized_xTB.columns = self.columns
        self.gridpoints_normalized = self.gridpoints_normalized_xTB
    @timer
    def run_Turbomole(self):
        x2t_dir = subprocess.run(['which', 'x2t'], stdout=subprocess.PIPE).stdout.decode().strip()
        define_dir = subprocess.run(['which', 'define'], stdout=subprocess.PIPE).stdout.decode().strip()
        ridft_dir = subprocess.run(['which', 'ridft'], stdout=subprocess.PIPE).stdout.decode().strip()
        promowa_dir = subprocess.run(['which', 'promowa'], stdout=subprocess.PIPE).stdout.decode().strip()
        path = str(os.getcwd())
        path_1 = path + '/1'
        path_2 = path + '/2'
        path_3 = path + '/3'
        with open('coord','w') as f:
            subprocess.run([x2t_dir, self.molecule], stdout=f)
        os.mkdir('1')
        os.system('mv -f coord 1')
        os.chdir(path_1)
        os.system(f'{define_dir} < ../def1 > out 2> out1')
        with open('control', 'r') as f:
            lines = f.readlines()
            lines.insert(-1, '$scfdenapproxl 0\n')
        with open('control', 'w') as f:
            f.writelines(lines)
        os.system(f'{ridft_dir} > out 2> out1')
        os.chdir(path)
        os.system('mkdir 2 3')
        self.gridpoints_Turbomole = []
        for i in range(len(self.gridpoints_filtered)):
            os.chdir(path_2)
            self.gridpoints_exporter(self.gridpoints_filtered.iloc[i:i + 1])
            with open('coord', 'w') as f:
                subprocess.run([x2t_dir, f'{self.molecule.split(".")[0]}_grids.xyz'], stdout=f)
            os.system(f'{define_dir} < ../def2 > out 2> out1')
            with open('control', 'r') as f:
                lines = f.readlines()
                lines.insert(-1, '$scfdenapproxl 0\n')
            with open('control', 'w') as f:
                f.writelines(lines)
            os.system(f'{ridft_dir} > out 2> out1')
            os.chdir(path_3)
            combined = pd.concat([self.molecule_coordinates,self.gridpoints_filtered.iloc[i:i + 1]],axis=0)
            self.gridpoints_exporter(combined)
            with open('coord', 'w') as f:
                subprocess.run([x2t_dir, f'{self.molecule.split(".")[0]}_grids.xyz'], stdout=f)
            os.system(f'{define_dir} < ../def3 > out 2> out1')
            with open('control', 'r') as f:
                lines = f.readlines()
                con1 = path_1 + '/control'
                con2 = path_2 + '/control'
                if i == 0:
                    lines.insert(-6, '$scfdenapproxl 0\n')
                    lines.insert(-6, '$subsystems\n')
                    lines.insert(-6, f'molecule#1 file={con1}\n')
                    lines.insert(-6, f'molecule#2 file={con2}\n')
            with open('control', 'w') as f:
                f.writelines(lines)
            subprocess.run([promowa_dir], stdout=subprocess.PIPE)
            with open('ridft.out','w') as f:
                result = subprocess.run([ridft_dir], stdout=f,stderr=subprocess.PIPE)
            if 'normally' in result.stderr.decode():
                    with open('ridft.out', 'r') as f:
                        lines = f.readlines()
                    with open('ridft.out', 'r') as f:
                        for index, line in enumerate(f):
                            if line.find('Total Interaction energy') != -1:
                                values = [line.split(' ')[-3] for line in lines[index:index+12]]
                                del values[1]
            else:
                values = ['0']*10
            self.gridpoints_Turbomole.append(values)
            with open('energies_Turbomole.out','a') as f:
                f.writelines(values)
                f.write('\n')
        os.system(f'mv -f energies_Turbomole.out ../{self.molecule.split(".")[0]}_Turbomole_Energies.out')
        self.gridpoints_Turbomole = pd.DataFrame(self.gridpoints_Turbomole,columns=['Eint_total','Eint_el','Eint_nuc','Eint_1e','Eint_2e','Eint_ex_rp','Eint_ex','Eint_rp','Eint_orb','Eint_cor','Eint_disp'],dtype=float)
        os.chdir(path)
        os.system('rm -rf 1 2 3')
        return self.gridpoints_Turbomole
            
          
         
                
                    















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






