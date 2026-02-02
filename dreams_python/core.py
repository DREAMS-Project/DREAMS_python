'''
Read functions for most of the data products from the DREAMS simulation suites

As of 02/02/2026, this has been tested and verifed to work (for the most part) on
- MW_Zooms, WDM, SB4
- MW_Zooms, CDM, SB5
- varied_mass, CDM, SB6
- varied_mass, CDM, SB9

Written by Alex M. Garcia (alexgarcia@virginia.edu) with thanks to Jonah Rose and
Ilem Leisher for providing some functions that have been adapted here.
'''
import h5py
import sys, os
import numpy as np
from pathlib import Path

class DREAMS:
    def __init__( self, base_path, suite='varied_mass', DM_type='CDM', sobol_number=6,
                  box_or_run='box', verbose=False,
                  layout=None ):
        '''Read Function for the DREAMS Cosmological Simulations

        Inputs:
        - base_path: base_path location where the simulation data lives
        - suite: which suite you're working with (e.g., "MW_Zooms", "varied_mass", "dwarfs")
        - DM_type: which DM type your simulation is (e.g., "CDM", "WDM", "SIDM")
        - sobol_number: how many parameters are varied
        (Optional)
        - box_or_run: specify whether your sims are named box or run
        - verbose: print helpful comments along the way
        - layout: helping in specifying the directory structure (see also self.set_path() )
        
        Example:
        import dreams_python

        mw_wdm = dreams_python.DREAMS( '/base/path/', suite='MW_Zooms', DM_type='WDM', sobol_number=4  )
        '''
        self.base_path    = base_path
        self.box_or_run   = box_or_run
        self.dm_type      = DM_type
        self.suite        = suite
        self.sobol_number = sobol_number
        self._verbose     = verbose

        self.layout = layout or { ## default paths
            "FOF_Subfind":     "{base}/FOF_Subfind/{dm}/{suite}/SB{sb}/run_{run}/fof_subhalo_tab_{snap}.hdf5",
            "SubLink":         "{base}/FOF_Subfind/{dm}/{suite}/SB{sb}/run_{run}/tree_extended.hdf5",
            "Sims":            "{base}/Sims/{dm}/{suite}/SB{sb}/run_{run}/snap_{snap}.hdf5",
            "Rockstar":        "{base}/Rockstar/{dm}/{suite}/SB{sb}/run_{run}/out_{snap}.list",
            "Parameters":      "{base}/Parameters/{dm}/{suite}/{fname}",
            "ConsistentTrees": "{base}/Rockstar/{dm}/{suite}/SB{sb}/run_{run}/tree_0_0_0.dat",
        }

        if self._verbose:
            print(f'Working with {DM_type} {suite} SB{sobol_number}')

        self.CT_COLUMNS = [
            'scale','id','desc_scale','desc_id','num_prog','pid','upid','desc_pid',
            'phantom','sam_Mvir','Mvir','Rvir','rs','vrms','mmp',
            'scale_of_last_MM','vmax','x','y','z','vx','vy','vz',
            'Jx','Jy','Jz','Spin','Breadth_first_ID','Depth_first_ID',
            'Tree_root_ID','Orig_halo_ID','Snap_idx',
            'Next_coprogenitor_depthfirst_ID',
            'Last_progenitor_depthfirst_ID',
            'Last_mainleaf_depthfirst_ID',
            'Tidal_Force','Tidal_ID','Rs_Klypin',
            'Mvir_all','M200b','M200c','M500c','M2500c',
            'Xoff','Voff','Spin_Bullock',
            'b_to_a','c_to_a','Ax','Ay','Az',
            'b_to_a500c','c_to_a500c','Ax500c','Ay500c','Az500c',
            'TU','M_pe_Behroozi','M_pe_Diemer',
            'Type','SM','Gas','BH_Mass'
        ]

    #############
    ##  Paths  ##
    #############

    def set_path(self, file_type, template):
        '''
        Set the layout for the simulation output directories

        Inputs:
        - file_type: any of ["Sims", "FOF_Subfind", "SubLink", "Rockstar", "Parameters", "ConsistentTrees"] and "*_Nbody" versions
        - template: fake f-string formatting for where data lives
            + acceptable key words:
                ~ base: base path of simulations
                ~ dm: dark matter type of simulations
                ~ suite: which suite type
                ~ sb: sobol number
                ~ run: simulation number
                ~ snap: simulation snapshot
                ~ fname: name of file (used internally for Parameters file)
        
        Example:
        dreams.set_path("Sims", "/scratch/{dm}/{suite}/SB{sb}")
        '''
        self.layout[file_type] = template
    
    def _resolve_dir(self, file_type, run, snap, DMO=False):
        '''Internal function used to get path'''
        sb = f"{self.sobol_number}_Nbody" if DMO else self.sobol_number ## default behavior

        this_key_count = 0
        layout_keys = self.layout.keys() 
        for key in layout_keys:
            if file_type in key:
                this_key_count += 1
        
        if DMO and this_key_count == 2:
            file_type = file_type + "_Nbody"

        template = self.layout[file_type]
        path = template.format(
            base=self.base_path,
            dm=self.dm_type,
            suite=self.suite,
            sb=f"{sb}",
            run=f"{run}",
            snap=f"{snap:03d}" if "Rockstar" not in file_type else f"{snap}",
            fname=run ## kinda hacky
        )
        return Path(path)

    def _check_path(self, path, type_file, run, snap=-1):
        '''Internal check of path existence'''
        if not os.path.exists(path):
            if self._verbose:
                print(f'{path} does not exist')
            if snap >= 0:
                raise(FileNotFoundError(f'{type_file} for {self.box_or_run} {run} at snap {snap} does not exist'))
            else:
                raise(FileNotFoundError(f'{type_file} for {self.box_or_run} {run} does not exist'))

    ######################
    ##  Read Functions  ##
    ######################
    
    def read_group_catalog(self, run, snap, keys=[], DMO=False):
        ''' 
        Read in Subfind group catalogs

        Inputs:
        - run: simulation number
        - snap: simulation snapshot
        (Optional)
        - keys: keywords to load in
        - DMO: toggle for Nbody versions

        Returns:
        - dict containing group catalog for specified simulation with specified keys
        '''
        path = self._resolve_dir("FOF_Subfind", run, snap, DMO)
        self._check_path(path, 'Group Catalog', run, snap)
        
        cat = dict()
        with h5py.File(path) as ofile:
            if len(keys) == 0:
                cat_keys = ofile.keys()
                for cat_key in cat_keys:
                    if len(ofile[cat_key]) == 0:
                        continue
                    keys += list(ofile[cat_key].keys())

            for key in keys:
                if 'Group' in key:
                    cat[key] = np.array(ofile[f'Group/{key}'])
                if 'Subhalo' in key:
                    cat[key] = np.array(ofile[f'Subhalo/{key}'])
        return cat
    
    def read_snapshot(self, run, snap, part_types=[], keys=[], DMO=False):
        ''' 
        Read in arepo particle data

        Inputs:
        - run: simulation number
        - snap: simulation snapshot
        (Optional)
        - part_types: particle types to load in (0->gas, 1->dm high res, 2->dm low res, 4->stars, 5->BHs)
        - keys: keywords to load in
        - DMO: toggle for Nbody versions

        Returns:
        - dict containing particle catalog for specified simulation with specified keys
        '''
        if type(part_types) == type(0):
            part_types = [part_types]
        if len(part_types) == 0:
            part_types = [0, 1, 2, 4, 5]
            if DMO:
                part_types = [1, 2]

        if DMO and any(pt in [0, 4, 5] for pt in part_types):
            raise KeyError("Cannot load baryons in DMO simulation")
        
        if len(keys) > 0:
            tmp_keys = []
            for key in keys:
                for pt in part_types:
                    if f'PartType{pt}' in key:
                        tmp_keys.append(key)
                    else:
                        tmp_keys.append( f'PartType{pt}/{key}' )
            keys = tmp_keys

        path = self._resolve_dir("Sims", run, snap, DMO)
        self._check_path(path, 'Snapshot', run, snap)

        cat = dict()
        with h5py.File(path) as ofile:
            if len(keys) == 0:
                for pt in part_types:
                    try:
                        cat_keys = ofile[f'PartType{pt}'].keys()
                    except KeyError:
                        continue
                    for cat_key in cat_keys:
                        if len(ofile[f'PartType{pt}/{cat_key}']) == 0:
                            continue
                        keys.append(f'PartType{pt}/{cat_key}')

                    if pt == 1:
                        keys.append('PartType1/Masses')
            
            for key in keys:
                if key == 'PartType1/Masses':
                    cat[key] = np.ones(ofile['PartType1/ParticleIDs'].shape)*ofile['Header'].attrs['MassTable'][1]
                else:
                    try:
                        cat[key] = np.array(ofile[key])
                    except KeyError as e:
                        continue
        return cat

    def read_rockstar(self, run, snap, keys=[], DMO=False):
        ''' 
        Read in Rockstar catalogs

        Inputs:
        - run: simulation number
        - snap: simulation snapshot
        (Optional)
        - keys: keywords to load in
        - DMO: toggle for Nbody versions

        Returns:
        - dict containing Rockstar catalog for specified simulation with specified keys
        '''
        path = self._resolve_dir("Rockstar", run, snap, DMO)
        self._check_path(path, 'Rockstar Catalog', run, snap)

        output = dict()
        with open(path, 'r') as f:
            data   = np.genfromtxt(f, names=True)
            if len(keys) == 0:
                keys = data.dtype.names
            for key in keys:
                if key not in data.dtype.names:
                    raise KeyError(f'{key} not in Rockstar Catalogs')
                output[key] = np.array(data[key])
        
        return output
    
    def read_param_file(self, fname):
        ''' 
        Read in DREAMS parameter file

        Inputs:
        - fname: name of file (relative, not absolute... see self.set_path() )

        Returns:
        - array containing parameters of simulation
        - list containing header of parameters
        '''
        path = self._resolve_dir("Parameters", fname, -1, False)
        data = np.loadtxt(path)
        header = None
        with open(path, 'r') as f:
            header = f.readline().strip().replace('#','')
        return data, header.split()

    def read_sublink_cat(self, run, keys=[], DMO=False):
        ''' 
        Read in SubLink merger trees

        Inputs:
        - run: simulation number
        (Optional)
        - keys: keywords to load in
        - DMO: toggle for Nbody versions

        Returns:
        - dict containing SubLink tree file for specified simulation with specified keys 
            Note that this is the *entire* file, not a specific tree
        '''
        path = self._resolve_dir("SubLink", run, -1, DMO)
        self._check_path(path, 'Sublink Catalog', run)
        
        cat = dict()
        with h5py.File(path, 'r') as ofile:
            if len(keys) == 0:
                keys = ofile.keys()
            for key in keys:
                cat[key] = np.array(ofile[key])
        return cat

    def read_consistent_trees(self, run, keys=[], DMO=False):
        ''' 
        Read in Consistent Trees merger trees

        Inputs:
        - run: simulation number
        (Optional)
        - keys: keywords to load in
        - DMO: toggle for Nbody versions

        Returns:
        - dict containing Consistent Trees file for specified simulation with specified keys 
            Note that this is the *entire* file, not a specific tree
        '''
        path = self._resolve_dir("ConsistentTrees", run, -1, DMO)
        self._check_path(path, 'Consistent Trees Catalog', run)

        lines = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                if len(line.split()) == len(self.CT_COLUMNS):
                    lines.append(line)
    
        data = np.fromstring(
            ''.join(lines),
            sep=' '
        ).reshape(-1, len(self.CT_COLUMNS))
    
        structured = np.zeros(len(data), dtype=[(c, 'f8') for c in self.CT_COLUMNS])
        for i, c in enumerate(self.CT_COLUMNS):
            structured[c] = data[:, i]
    
        if len(keys) == 0:
            keys = self.CT_COLUMNS
    
        return {key: structured[key] for key in keys}

    def read_header(self, run, snap, DMO=False):
        ''' 
        Read in header from arepo snapshot files

        Inputs:
        - run: simulation number
        - snap: simulation snapshot
        (Optional)
        - DMO: toggle for Nbody versions

        Returns:
        - dict containing all header attributes 
        '''
        attrs = {}
        path = self._resolve_dir("Sims", run, snap, DMO)
        self._check_path(path, 'Snapshot', run, snap)

        with h5py.File(path, 'r') as f:
            hdr = f['Header']
            for key in hdr.attrs:
                attrs[key] = hdr.attrs[key]
        return attrs
    
    ########################
    ##  Header Shortcuts  ##
    ########################
        
    def get_scf(self, run, snap):
        '''Get scale factor at specified simulation snapshot'''
        scf = None
        DMO = False ## should be same dmo and hydro
        path = self._resolve_dir("Sims", run, snap, DMO)
        self._check_path(path, 'Snapshot', run, snap)
        
        with h5py.File(path, 'r') as f:
            scf=f['Header'].attrs['Time']
        return scf

    def get_h(self, run, snap): 
        '''Get hubble factor at specified simulation snapshot'''
        h = None
        DMO = False ## should be same dmo and hydro
        path = self._resolve_dir("Sims", run, snap, DMO)
        self._check_path(path, 'Snapshot', run, snap)
        
        with h5py.File(path, 'r') as f:
            h = f['Header'].attrs['HubbleParam']
        return h

    def get_box_size(self, run, snap):
        '''Get box size at specified simulation snapshot'''
        box_size = None
        DMO = False ## should be same dmo and hydro
        path = self._resolve_dir("Sims", run, snap, DMO)
        self._check_path(path, 'Snapshot', run, snap)
        
        with h5py.File(path, 'r') as f:
            box_size=f['Header'].attrs['BoxSize']
        return box_size
    
    def get_high_res_dm_mass(self, run, snap):
        '''Get high res dm mass resolution at specified simulation snapshot'''
        hdr = self.get_header(run, snap)
        return hdr['MassTable'][1]

    #####################
    ##  Contamination  ##
    #####################
    
    def get_contamination_dm(self, group_catalog):
        '''
        Get dark matter contamination fraction for all halos in group catalog

        Inputs:
        - group_catalog: Subfind catalog (must contain at least "GroupMassType")

        Returns:
        - array of contamination fractions
        '''
        keys = group_catalog.keys()
        try:
            assert("GroupMassType" in keys)
        except AssertionError:
            raise(AssertionError("Need to Load in GroupMassType into group catalog"))

        masses = group_catalog['GroupMassType']
        return masses[:, 2] / np.sum(masses, axis=1)

    def get_contamination_baryon(self, run, snap, fof_idx=-1, subhalo_idx=-1):
        '''
        Get gas contamination fraction for specific halo/subhalo

        Inputs:
        - run: simulation number
        - snap: simulation snapshot
        (Optional)
        - fof_idx: index of halo to target 
        - subhalo_idx: index of subhalo to target (unused if fof_idx used)

        Returns:
        - float of contamination fraction
        '''
        if fof_idx > -1 and subhalo_idx > -1:
            raise ValueError('Please specify only fof_idx or subhalo_idx, not both')
            
        part_types = [0, 4]
        keys = ['PartType0/AllowRefinement',
                'PartType4/Masses']

        if fof_idx > -1:
            prt_cat = self.load_single_halo(run, snap, fof_idx, part_types, keys)
        elif subhalo_idx > -1:
            prt_cat = self.load_single_subhalo(run, snap, sub_idx, part_types, keys)

        refined = prt_cat['PartType0/AllowRefinement']
        low_res_gas = refined == 0

        gas_contam = sum(low_res_gas) / len(refined)
        return gas_contam

    ###################################
    ##  Identify Target of Interest  ##
    ###################################
        
    def get_target_fof_index(self, run, snap, target_mass, max_dm=0.25, max_contam=0.25, DMO=False):
        '''
        Target a specific mass halo with specific contamination based on the Subfind catalogs

        Inputs:
        - run: simulation number
        - snap: simulation snapshot
        - target_mass: (log) mass of halo of interest
        (Optional)
        - max_dm: max tolerance in delta mass 
        - max_contam: max tolerance in contamination
        - DMO: toggle for Nbody versions (note that this is not guarenteed to give you the same halo as the hydro version)

        Returns:
        - integer of fof index corresponding to your target
        '''
        if DMO:
            print('!!! Warning !!! Selecing a DMO simulation will work, but it is not guarenteed to be the correct halo')
            print('!!! Warning !!! It is recommended you use "match_halo_hydro_dmo" instead')
        grp_cat = self.read_group_catalog(run, snap, keys=['GroupMassType'], DMO=DMO)
        h = self.get_h(run, snap)
        
        grp_masses = grp_cat['GroupMassType'] * 1.00E+10 / h
        ids = np.arange(len(grp_masses))
        
        log_grp_mass = np.log10(np.sum(grp_masses,axis=1) + 1e-16)

        dm = np.abs(log_grp_mass - target_mass)

        grp_contam = self.get_contamination_dm(grp_cat)

        dm_tolerance     = max_dm 
        contam_tolerance = max_contam
        tolerable_grps   = (dm < dm_tolerance) & (grp_contam < contam_tolerance)
        
        if sum(tolerable_grps) == 0:
            print("0 groups found")
            return -1

        log_grp_mass = log_grp_mass[tolerable_grps]
        dm           = dm[tolerable_grps]
        grp_contam   = grp_contam[tolerable_grps]
        ids          = ids[tolerable_grps]
    
        EPS    = 1e-5
        scores = dm*10 + np.log10(grp_contam + EPS)
    
        winner = np.argmin(scores)

        if self._verbose:
            this_mass_res   = np.log10(self.get_high_res_dm_mass(run,snap) * 1.00e+10/h)
            this_gas_contam = self.get_contamination_baryon(run, snap, ids[winner])

            dmo_tag = ' (DMO)' if DMO else ''
            
            print(f'!!! Found Target in Simulation {run}{dmo_tag} at snapshot {snap} !!!')
            print(f'Your halo has:')
            print(f'\tHalo Mass              : {log_grp_mass[winner]:0.3f} [log Msun]')
            print(f'\tHR DM Mass Resolution  : {this_mass_res:0.3f} [log Msun]')
            print(f'\tDM Contamination       : {grp_contam[winner]*100:0.3f}%')
            if not DMO:
                print(f'\tGas Contamination      : {this_gas_contam*100:0.3f}%')
            print('')
        return ids[winner]

    def get_target_central_subhalo_index(self, run, snap, target_mass, max_dm=0.25, max_contam=0.25, DMO=False):
        '''
        Target the central subhalo of a specific halo mass with specific contamination based on the Subfind catalogs

        Inputs:
        - run: simulation number
        - snap: simulation snapshot
        - target_mass: (log) mass of halo of interest
        (Optional)
        - max_dm: max tolerance in delta mass 
        - max_contam: max tolerance in contamination
        - DMO: toggle for Nbody versions (note that this is not guarenteed to give you the same halo as the hydro version)

        Returns:
        - integer of subhalo id corresponding to your target
        '''
        h = self.get_h(run, snap)
        grp_cat = self.read_group_catalog(run, snap, keys=['GroupFirstSub'])
        fof_idx = self.get_target_fof_index(run, snap, target_mass, max_dm=max_dm, max_contam=max_contam, DMO=DMO)

        return grp_cat['GroupFirstSub'][fof_idx]

    def get_target_rockstar_index(self, run, snap, target_mass, max_dm=0.25, DMO=False,
                                  _rockstar_units=1e3):
        '''
        Target a specific mass halo with specific contamination based on the Rockstar catalogs
        (note that this assumes you also have Subfind catalogs to verify the target)

        Inputs:
        - run: simulation number
        - snap: simulation snapshot
        - target_mass: (log) mass of halo of interest
        (Optional)
        - max_dm: max tolerance in delta mass
        - DMO: toggle for Nbody versions (note that this is not guarenteed to give you the same halo as the hydro version)
        - _rockstar_units: convert rockstar units (Mpc) to Subfind units (kpc)

        Returns:
        - integer of fof index corresponding to your target
        '''
        rockstar_cat = self.read_rockstar(run, snap)
        
        ## get close matches in halo mass
        ids = rockstar_cat['ID'].astype(int)
        targets = np.abs(np.log10(rockstar_cat['Mvir']) - target_mass)
        potential = targets < max_dm
        
        if self._verbose:
            print(f'Found {sum(potential)} targets within {max_dm} dex')

        if sum(potential) == 0:
            print(f'0 targets within {max_dm} dex of {target_mass}')
            return -1

        grp_cat = self.read_group_catalog(run, snap, keys=['GroupPos'], DMO=DMO) ## load FoF to verify
        if DMO:
            _, fof_idx = self.match_halo_hydro_dmo(run, snap, target_mass)
        else:
            fof_idx = self.get_target_fof_index(run, snap, target_mass) 
        fof_pos = grp_cat['GroupPos'][fof_idx]
        
        ## compare position to FoF
        ids = ids[potential]
        x, y, z = rockstar_cat['X'][potential], rockstar_cat['Y'][potential], rockstar_cat['Z'][potential]
    
        rs_pos = np.vstack([x, y, z]).T * _rockstar_units ## Assuming rockstar in Mpc

        rockstar_id = ids[np.argmin(np.linalg.norm(fof_pos - rs_pos, axis=1))]
        
        return np.where(rockstar_cat["ID"] == rockstar_id)[0][0]

    ####################
    ##  Merger Trees  ##
    ####################
    
    def get_sublink_mpb(self, run, snap, subhalo_idx=-1, DMO=False):
        '''
        Get the main progenitor branch from SubLink catalogs

        Inputs:
        - run: simulation number
        - snap: simulation snapshot
        - subhalo_idx: root node galaxy to target (note: *not* halo, subhalo)
        (Optional)
        - DMO: toggle for Nbody versions

        Returns:
        - dict with SubLink main progenitor branch
        '''
        sublink_tree = self.read_sublink_cat(run, DMO=DMO)
        
        if subhalo_idx > -1:
            mask = ( (sublink_tree['SnapNum'] == snap) & (sublink_tree['SubhaloID'] == subhalo_idx) )
            match = np.where( mask )[0]
            if match.size == 0:
                print(f'!!! Warning !!! Subhalo {subhalo_idx} tree not found')
                return {}
            target = match[0]
        else:
            raise KeyError(f'Please pass subhalo_idx')

        cat = dict()
        for key in sublink_tree.keys(): ## add target info
            cat[key] = [sublink_tree[key][target]]

        fpID = sublink_tree['FirstProgenitorID'][target] ## get target's first progenitor
        while fpID != -1: ## add progenitor info
            for key in sublink_tree.keys():
                cat[key] += [sublink_tree[key][fpID]]
            fpID = sublink_tree['FirstProgenitorID'][fpID]

        cat = {
            key: np.array(cat[key]) for key in cat
        }
            
        return cat

    def get_consistent_trees_mpb(self, run, snap, subhalo_idx=-1, DMO=False):
        '''
        Get the main progenitor branch from Consistent Trees catalogs

        Inputs:
        - run: simulation number
        - snap: simulation snapshot
        - subhalo_idx: root node galaxy to target (from Rockstar)
        (Optional)
        - DMO: toggle for Nbody versions

        Returns:
        - dict with Consistent Trees main progenitor branch
        '''
        if subhalo_idx == -1:
            raise KeyError(f'Please pass subhalo_idx')
        
        rockstar_cat = self.read_rockstar(run, snap, DMO=DMO)
        Mvir_rs = rockstar_cat['Mvir'][subhalo_idx]
        
        ct = self.read_consistent_trees(run, DMO=DMO)

        ## Find where target lives in Consistent Tree catalog
        scf        = self.get_scf(run, snap)
        this_snap  = int(ct['Snap_idx'][np.isclose(ct['scale'], scf)][0])
        ids        = np.arange(len(ct['Snap_idx'])).astype(int)
        snap_mask  = (ct["Snap_idx"] == this_snap)    
        mass_match = np.argmin(np.abs( np.log10(ct['Mvir'][snap_mask]) - np.log10(Mvir_rs) ))
        ct_id      = ids[snap_mask][mass_match]

        ## Walk the main progenitor branch
        branch = []
        
        root_id = ct['id'][ct_id]
        idx = np.where( ct['id'] == root_id )[0][0]
        while True:
            branch.append(idx)
            progs = np.where(ct['desc_id'] == ct['id'][idx])[0]
            if len(progs) == 0:
                break
            idx = progs[ np.argmax( ct['Mvir'][progs] ) ]

        ## Create out structure
        mpb = {}
        for node in branch:
            for key in ct.keys():
                if key not in mpb:
                    mpb[key] = [ct[key][node]]
                else:
                    mpb[key] += [ct[key][node]]
        for key in mpb.keys():
            mpb[key] = np.array( mpb[key] )
        return mpb

    def get_sublink_tree(self, run, snap, subhalo_idx=-1, DMO=False):
        '''
        Get the full merger history from SubLink catalogs

        Inputs:
        - run: simulation number
        - snap: simulation snapshot
        - subhalo_idx: root node galaxy to target (note: *not* halo, subhalo)
        (Optional)
        - DMO: toggle for Nbody versions

        Returns:
        - dict with full SubLink merger history
        '''
        sublink_tree = self.read_sublink_cat(run, DMO=DMO)    
        snap_overlap = (sublink_tree['SnapNum'] == snap)

        if sum(snap_overlap) == 0:
            raise ValueError(f'SubLink has no subhalos at snap {snap}')
        
        if subhalo_idx > -1:
            mask = ( (sublink_tree['SnapNum'] == snap) & (sublink_tree['SubhaloID'] == subhalo_idx) )
            match = np.where( mask )[0]
            if match.size == 0:
                print(f'!!! Warning !!! Subhalo {subhalo_idx} tree not found')
                return {}
            target = match[0]
        else:
            raise KeyError(f'Please pass either fof_idx or subhalo_idx')

        cat = dict()
        for key in sublink_tree.keys(): ## add target info
            cat[key] = [sublink_tree[key][target]]

        fpID = sublink_tree['FirstProgenitorID'][target] ## get target's first progenitor (at snap - 1)
        npID = sublink_tree['NextProgenitorID'][target]  ## get target's next progentiro (at current snap)

        ids_to_walk = [target]
        while npID != -1:
            for key in sublink_tree.keys():
                cat[key] += [sublink_tree[key][fpID]]
            ids_to_walk.append(target)
            npID = sublink_tree['NextProgenitorID'][target]

        def walk_tree(ID, cat, sublink_tree):
            fpID = sublink_tree['FirstProgenitorID'][ID]

            while fpID != -1:
                for key in sublink_tree.keys():
                    cat[key] += [sublink_tree[key][fpID]]

                npID = sublink_tree['NextProgenitorID'][fpID]
                while npID != -1:
                    for key in sublink_tree.keys():
                        cat[key] += [sublink_tree[key][npID]]
                    walk_tree(npID, cat, sublink_tree)

                    npID = sublink_tree['NextProgenitorID'][npID]
                
                fpID = sublink_tree['FirstProgenitorID'][fpID]
            
            return

        for ID in ids_to_walk:
            walk_tree(ID, cat, sublink_tree)

        cat = {
            key: np.array(cat[key]) for key in cat
        }
        
        return cat

    def get_consistent_tree(self, run, snap, subhalo_idx=-1, DMO=False):
        '''
        Get the full merger history from Consistent Trees catalogs

        Inputs:
        - run: simulation number
        - snap: simulation snapshot
        - subhalo_idx: root node galaxy to target (from Rockstar)
        (Optional)
        - DMO: toggle for Nbody versions

        Returns:
        - dict with full Consistent Trees merger history
        '''
        if subhalo_idx == -1:
            raise KeyError(f'Please pass subhalo_idx')
        
        rockstar_cat = self.read_rockstar(run, snap, DMO=DMO)
        Mvir_rs = rockstar_cat['Mvir'][subhalo_idx]
        
        ct = self.read_consistent_trees(run, DMO=DMO)

        ## Find where target lives in Consistent Tree catalog
        scf        = self.get_scf(run, snap)
        this_snap  = int(ct['Snap_idx'][np.isclose(ct['scale'], scf)][0])
        ids        = np.arange(len(ct['Snap_idx'])).astype(int)
        snap_mask  = (ct["Snap_idx"] == this_snap)    
        mass_match = np.argmin(np.abs( np.log10(ct['Mvir'][snap_mask]) - np.log10(Mvir_rs) ))
        ct_id      = ids[snap_mask][mass_match]

        def walk_tree(ID, visited, ct):
            '''Recursively walk consistent tree'''
            if not isinstance(ID, int):
                ID = int(ID)
            if ID in visited:
                return
        
            visited.add(ID)
        
            progs = np.where(ct['desc_id'] == ct['id'][ID])[0]
            assert( len(progs) == ct['num_prog'][ID] )
            for p in progs:
                walk_tree(p, visited, ct)

        ## Walk the whole tree
        root_id = ct['id'][ct_id]
        idx = np.where( ct['id'] == root_id )[0][0]
        
        visited = set()
        walk_tree(idx, visited, ct)

        nodes = np.array(sorted(visited), dtype=int)
        
        tree = {}
        for node in nodes:
            for key in ct.keys():
                if key not in tree:
                    tree[key] = [ct[key][node]]
                else:
                    tree[key].append(ct[key][node])
        for key in tree:
            tree[key] = np.array(tree[key])
    
        return tree

    #################################################
    ##  Load in Particle Data for specific target  ##
    #################################################
    
    def load_single_halo(self, run, snap, fof_idx=-1, part_types=[], keys=[], DMO=False):
        '''
        Load particles for a single FoF halo from arepo outputs

        Inputs:
        - run: simulation number
        - snap: simulation snapshot
        - fof_idx: halo index to load
        (Optional)
        - part_types: particle types to load, all if left empty
        - keys: keys to load, all if left empty
        - DMO: toggle for Nbody versions

        Returns:
        - dictionary with particle information for all particles within halo
        '''
        if fof_idx == -1:
            raise ValueError('Please specify halo to load')

        if type(part_types) == type(0):
            part_types = [part_types]
        if len(part_types) == 0:
            part_types = [0, 1, 2, 4, 5]
            if DMO:
                part_types = [1, 2]

        if DMO and any(pt in [0, 4, 5] for pt in part_types):
            raise KeyError("Cannot load baryons in DMO simulation")
        
        if len(keys) > 0:
            tmp_keys = []
            for key in keys:
                for pt in part_types:
                    if f'PartType' in key:
                        tmp_keys.append(key)
                    else:
                        tmp_keys.append( f'PartType{pt}/{key}' )
            keys = tmp_keys
        
        grp_cat = self.read_group_catalog(run, snap, keys=['GroupFirstSub', 'GroupNsubs'], DMO=DMO)
        
        first_sub = grp_cat['GroupFirstSub'][fof_idx]
        nsubs     = grp_cat['GroupNsubs'][fof_idx]
    
        all_parts = []
    
        for sub in range(first_sub, first_sub + nsubs):
            sub_cat = self.load_single_subhalo(run, snap, sub_idx=sub, part_types=part_types, keys=keys, DMO=DMO)
            all_parts.append(sub_cat)

        if len(keys) == 0:
            all_keys = set()
            for gal in all_parts:
                for key in gal.keys():
                    all_keys.add(key)
            keys = list(all_keys)
        
        merged = {}
        for key in keys:
            arrays = []
            for p in all_parts:
                if key not in p:
                    continue
                arr = p[key]
                # Ensure empty arrays have the correct number of dimensions
                if arr.size == 0:
                    # Determine correct ndim from first non-empty array
                    for ref in all_parts:
                        if key in ref and ref[key].size > 0:
                            ndim = ref[key].ndim
                            shape = (0,) + ref[key].shape[1:]
                            arr = np.empty(shape, dtype=ref[key].dtype)
                            break
                arrays.append(arr)
            if len(arrays) == 0:
                continue
            merged[key] = np.concatenate(arrays, axis=0)
        
        return merged

    def load_single_subhalo(self, run, snap, sub_idx=-1, part_types=[], keys=[], DMO=False):
        '''
        Load particles for a single subhalo from arepo outputs

        Inputs:
        - run: simulation number
        - snap: simulation snapshot
        - sub_idx: halo index to load
        (Optional)
        - part_types: particle types to load, all if left empty
        - keys: keys to load, all if left empty
        - DMO: toggle for Nbody versions

        Returns:
        - dictionary with particle information for all particles within subhalo
        '''
        if sub_idx == -1:
            raise ValueError("Please specify subhalo to load")
    
        if type(part_types) == type(0):
            part_types = [part_types]
        if len(part_types) == 0:
            part_types = [0, 1, 2, 4, 5]
            if DMO:
                part_types = [1, 2]
    
        if DMO and any(pt in [0, 4, 5] for pt in part_types):
            raise KeyError("Cannot load baryons in DMO simulation")
    
        path = self._resolve_dir("Sims", run, snap, DMO)
        self._check_path(path, 'Snapshot', run, snap)
    
        grp_cat = self.read_group_catalog(run, snap, keys=['SubhaloLenType'], DMO=DMO)
        lens = grp_cat['SubhaloLenType'][sub_idx]
    
        grp_all = self.read_group_catalog(run, snap, keys=['SubhaloLenType'], DMO=DMO)
        offsets = np.zeros_like(lens)
        for pt in part_types:
            offsets[pt] = np.sum(grp_all['SubhaloLenType'][:sub_idx, pt])
    
        # Load data
        cat = {}
        with h5py.File(path, 'r') as f:
            if len(keys) == 0:
                keys = []
                for pt in part_types:
                    if f'PartType{pt}' not in f:
                        continue
                    keys += [f'PartType{pt}/{k}' for k in f[f'PartType{pt}'].keys()]
                    if pt == 1:
                        keys.append('PartType1/Masses')
            
            for key in keys:
                pt = int(key.split("/")[0][-1])
                start = offsets[pt]
                length = lens[pt]
                if length == 0:
                    cat[key] = np.array([])
                elif key == 'PartType1/Masses':
                    mass = f['Header'].attrs['MassTable'][pt]
                    cat[key] = np.ones(length) * mass
                else:
                    cat[key] = f[key][start:start + length]
    
        return cat

    #######################
    ##  Match DMO/Hydro  ##
    #######################
        
    def match_halo_hydro_dmo(self, run, snap, target_mass, _full_search=True,
                             mass_tolerance=0.5, contamination_tolerance=0.2):
        '''
        Match a hydro target to its corresponding dark matter only counterpart

        Inputs:
        - run: simulation number
        - snap: simulation snapshot
        - target_mass: (log) mass of halo of interest
        (Optional)
        - _full_seach: see Note below
        - mass_tolerance: tolerance of mass matches for _full_search
        - contamination_tolerance: tolerance of contamination for _full_search

        Note:
        _full_search = True 
            compares against every single halo in the box regardless of mass or contamination
        _full_search = False 
            (not recommended, but faster)
            assumes that your dmo target is within mass_tolerance and contamination_tolerance
        '''
        fof_idx = self.get_target_fof_index(run, snap, target_mass)

        h = self.get_h(run, snap)
        
        dm_prts = [1, 2]
        keys    = ['ParticleIDs']
    
        prt_cat_hydro = self.load_single_halo(run, snap, fof_idx=fof_idx, part_types=dm_prts, keys=keys)

        all_hydro_ids = np.concatenate([ prt_cat_hydro['PartType1/ParticleIDs'], prt_cat_hydro['PartType2/ParticleIDs'] ])
        
        prt_cat_dmo = self.read_snapshot(run, snap, part_types=dm_prts, keys=keys, DMO=True)
        grp_cat_dmo = self.read_group_catalog(run, snap, keys=['GroupMassType'], DMO=True)

        dmo_masses = np.log10(np.sum(grp_cat_dmo['GroupMassType'], axis=1) * 1.00E+10 / h)
        ids = np.arange(len(dmo_masses))
        if not _full_search:
            targets_within_tolerance = np.abs(dmo_masses - target_mass) < mass_tolerance

            if self._verbose:
                print(f'Found {sum(targets_within_tolerance)} targets within {mass_tolerance} dex...')
    
            if sum(targets_within_tolerance) == 0:
                raise ValueError(f'No targets within {mass_tolerance} dex of your halo. Maybe broaden this range?')
    
            ids = ids[targets_within_tolerance]
            contaminations = grp_cat_dmo['GroupMassType'][targets_within_tolerance,2] /\
                             np.sum(grp_cat_dmo['GroupMassType'][targets_within_tolerance], axis=1) 
    
            reasonable_contam = contaminations < contamination_tolerance
    
            if sum(reasonable_contam) == 0:
                raise ValueError(f'No targets with contamination < {contamination_tolerance}. Maybe broaden this range?')
                
            ids = ids[reasonable_contam]
            if self._verbose:
                print(f'Found {sum(reasonable_contam)} with contamination < {contamination_tolerance}...')

        
        scores = np.zeros(len(ids), dtype=float)
        if self._verbose:
            print(f'Scoring {len(ids)} targets for a match...\n')
        for index, this_id in enumerate(ids):
            single_dmo = self.load_single_halo(run, snap, fof_idx=this_id, part_types=dm_prts, keys=keys, DMO=True)

            if 'PartType1/ParticleIDs' not in single_dmo: ## if no high res particles this is not my target
                continue

            if 'PartType2/ParticleIDs' not in single_dmo:
                single_dmo_ids = single_dmo['PartType1/ParticleIDs']
            else:
                single_dmo_ids = np.concatenate([ single_dmo['PartType1/ParticleIDs'], single_dmo['PartType2/ParticleIDs'] ])

            common_ids = np.intersect1d(single_dmo_ids, all_hydro_ids, assume_unique=True)
            
            scores[index] = len(common_ids) / len(all_hydro_ids)
            if scores[index] > 0.7: ## if they're sufficiently good matches continue
                break

        if self._verbose:
            print(f'Hydro-DMO Matching Best Score: {scores[np.argmax(scores)]*100:0.3f}% overlap')
        
        return fof_idx, ids[np.argmax(scores)]
        
if __name__ == "__main__":
    print('Hello World!')