import numpy as np
import pandas as pd
from numpy import linspace
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
import os
import subprocess
import scipy.spatial.transform as sst
os.environ['OVITO_GUI_MODE'] = '1'
from ovito.io import *
from ovito.modifiers import *
from ovito.data import *
from ovito.pipeline import *
from ovito.vis import *
from ovito.qt_compat import QtCore
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
        self.gridpoints_filtered_1 = np.array([row for i, row in enumerate(gridpoints_coordinate) if i in filtered_index])
        self.dist_filtered_1 = np.array([row for i, row in enumerate(self.dist) if i in filtered_index])
        dist_filtered_min_1 = np.amin(self.dist_filtered_1, axis=1)
        filtered_index = np.where(dist_filtered_min_1 < self.sieve)[0]
        self.dist_filtered = np.array([row for i, row in enumerate(self.dist_filtered_1) if i in filtered_index])
        self.gridpoints_filtered = pd.DataFrame(np.array([row for i, row in enumerate(self.gridpoints_filtered_1) if i in filtered_index]))
        self.gridpoints_filtered.insert(0, 'atom_name', self.probe)
        self.gridpoints_filtered.columns = ['atom_name', 'X', 'Y', 'Z']
    @timer
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
                    rotated = np.column_stack((rotated,self.gridpoints_nomorlized_EDA[list(val)].to_numpy()))
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
                    data.particles_.particle_types_.type_by_name_(self.probe).radius = 0.15
                else:
                    data.particles_.particle_types_.type_by_name_(self.probe).radius = 0.05
            gridpoints_v.modifiers.append(modify_pipeline_input)
            data_1 = gridpoints_v.compute() # Evaluate new pipeline to gain access to visual elements associated with the imported data objects.
            data_1.particles.vis.radius = 0.2
            data_1.cell.vis.enabled = False
            del data_1 # Done accessing input DataCollection of pipeline.
            if len(val) != 0:
                top = self.gridpoints_nomorlized_EDA[eda_val].median() + self.gridpoints_nomorlized_EDA[eda_val].std()
                btm = self.gridpoints_nomorlized_EDA[eda_val].median() - self.gridpoints_nomorlized_EDA[eda_val].std()
                gridpoints_v.modifiers.append(ColorCodingModifier(
                    property = eda_val,
                    start_value = max(0,btm),
                    end_value =  min(1,top),
                    gradient = ColorCodingModifier.Rainbow()))
                mod2 = ConstructSurfaceModifier()
                mod2.radius = 5.5
                mod2.smoothing_level = 30
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
            molecule_v.remove_from_scene()
            gridpoints_v.remove_from_scene()
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
        self.gridpoints_EDA = []
        for self.i,gridpoint in enumerate(self.gridpoints_filtered.to_numpy()):
            os.chdir(path_2)
            with open('gridpoint.xyz','w') as f:
                f.write('1')
                f.write('\n\n')
                np.savetxt(f,self.gridpoints_filtered.iloc[self.i:self.i+1].to_numpy(),fmt='%s')
                f.close()
            subprocess.run([xtb_dir, 'gridpoint.xyz',f'--chrg {chrg}' ,'--lmo',f'--gfn {gfn}'],stdout=subprocess.DEVNULL)
            os.rename('xtblmoinfo','2')
            os.system('mv -f '+'2 '+ path_3)
            os.chdir(path_3)
            with open('int.txt','w') as f:
                subprocess.run([xtbiff_dir, '1','2' ,'-sp'],stdout=f)
            with open('int.txt', 'r') as f:
                lines = f.readlines()
                gridpoint_EDA =np.array([line.split(':')[1] for line in lines[-11:-1]]).astype(float)
            self.gridpoints_EDA.append(gridpoint_EDA)
        with open('int.txt', 'r') as f:
            lines = f.readlines()
            self.columns =np.array(["_".join((line.split(':')[0]).split()) for line in lines[-11:-1]]).tolist()
        self.gridpoints_EDA = pd.DataFrame(self.gridpoints_EDA)
        self.gridpoints_EDA.columns = self.columns
        os.chdir(path)
        os.system('rm -rf 1 2 3')
        scaler = MinMaxScaler()
        self.gridpoints_nomorlized_EDA =pd.DataFrame(scaler.fit_transform(self.gridpoints_EDA[self.columns]))
        self.gridpoints_nomorlized_EDA.columns = self.columns
    def xyz_exporter(self,axis,animation_speed,*val,vp=Viewport(type = Viewport.Type.Front,fov = 11,camera_pos = (0,0,0),camera_dir = (1,0,0))):
        self.gridpoints_visualizer(axis, animation_speed,0, 5, vp, [False,False,(1, 2),False], *val)
    @timer
    def anime_visualizer(self,axis,angle,fps,figsize:tuple,*val,mol=False,label='Eint_total,gas',vp=Viewport(type = Viewport.Type.Front,fov = 11,camera_pos = (0,0,0),camera_dir = (1,0,0))):
        self.xyz_exporter(axis,angle,*val)
        self.gridpoints_visualizer(0,3.6,0,fps,vp,[True,mol,figsize,True],*val,eda_val=label)
    @timer
    def image_visualizer(self,axis,angle,frame,figsize:tuple,*val,mol=True,label='Eint_total,gas',vp=Viewport(type = Viewport.Type.Front,fov = 11,camera_pos = (0,0,0),camera_dir = (1,0,0))):
        self.xyz_exporter(axis, angle, *val)
        self.gridpoints_visualizer(0,3.6,frame,1,vp,[True,mol,figsize,False],*val,eda_val=label)



