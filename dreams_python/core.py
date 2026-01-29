import h5py
import sys, os
import numpy as np

class DREAMS:
    def __init__( self, base_path, suite='varied_mass', DM_type='CDM', sobol_number=6, box_or_run='box', verbose=False ):
        self.base_path    = base_path
        self.box_or_run   = box_or_run
        self.dm_type      = DM_type
        self.suite        = suite
        self.sobol_number = sobol_number
        self._verbose     = verbose
    
        self.dir     = lambda file_type: f'{base_path}/{file_type}/{DM_type}/{suite}/SB{sobol_number}'
        self.dir_dmo = lambda file_type: f'{base_path}/{file_type}/{DM_type}/{suite}/SB{sobol_number}_Nbody'

        if self._verbose:
            print(f'Working with {DM_type} {suite} SB{sobol_number}')
    
    def check_path(self, path, type_file, run, snap=-1):
        if not os.path.exists(path):
            if self._verbose:
                print(f'{path} does not exist')
            if snap >= 0:
                raise(FileNotFoundError(f'{type_file} for {self.box_or_run} {run} at snap {snap} does not exist'))
            else:
                raise(FileNotFoundError(f'{type_file} for {self.box_or_run} {run} does not exist'))
    
    def read_group_catalog(self, run, snap, keys=[], DMO=False):
        path_func = self.dir
        if DMO:
            path_func = self.dir_dmo
            
        path = f"{path_func('FOF_Subfind')}/{self.box_or_run}_{run}/fof_subhalo_tab_{snap:03d}.hdf5"
        self.check_path(path, 'Group Catalog', run, snap)
        
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

        path_func = self.dir
        if DMO:
            path_func = self.dir_dmo
        
        path = f"{path_func('Sims')}/{self.box_or_run}_{run}/snap_{snap:03d}.hdf5"
        self.check_path(path, 'Snapshot', run, snap)

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

    def read_param_file(self, fname):
        path = f'{self.base_path}/Parameters/{self.dm_type}/{self.suite}/{fname}'
        data = np.loadtxt(path)
        header = None
        with open(path, 'r') as f:
            header = f.readline().strip().replace('#','')
        return data, header.split()

    def read_sublink_cat(self, run, keys=[], DMO=False):
        path_func = self.dir
        if DMO:
            path_func = self.dir_dmo
            
        path = f"{path_func('FOF_Subfind')}/{self.box_or_run}_{run}/tree_extended.hdf5"
        self.check_path(path, 'Sublink Catalog', run)
        
        cat = dict()
        with h5py.File(path, 'r') as ofile:
            if len(keys) == 0:
                keys = ofile.keys()
            for key in keys:
                cat[key] = np.array(ofile[key])
        return cat
    
    def get_scf(self, run, snap): ## should be same dmo and hydro
        scf = None
        path = f"{self.dir('Sims')}/{self.box_or_run}_{run}/snap_{snap:03d}.hdf5"
        self.check_path(path, 'Snapshot', run, snap)
        
        with h5py.File(path, 'r') as f:
            scf=f['Header'].attrs['Time']
        return scf

    def get_h(self, run, snap): ## should be same dmo and hydro
        h = None
        path = f"{self.dir('Sims')}/{self.box_or_run}_{run}/snap_{snap:03d}.hdf5"
        self.check_path(path, 'Snapshot', run, snap)
        
        with h5py.File(path, 'r') as f:
            h = f['Header'].attrs['HubbleParam']
        return h

    def get_box_size(self, run, snap): ## should be same dmo and hydro
        box_size = None
        path = f"{self.dir('Sims')}/{self.box_or_run}_{run}/snap_{snap:03d}.hdf5"
        self.check_path(path, 'Snapshot', run, snap)
        
        with h5py.File(path, 'r') as f:
            box_size=f['Header'].attrs['BoxSize']
        return box_size

    def get_header(self, run, snap, DMO=False):
        path_func = self.dir
        if DMO:
            path_func = self.dir_dmo
        
        attrs = {}
        path = f"{path_func('Sims')}/{self.box_or_run}_{run}/snap_{snap:03d}.hdf5"
        self.check_path(path, 'Snapshot', run, snap)

        with h5py.File(path, 'r') as f:
            hdr = f['Header']
            for key in hdr.attrs:
                attrs[key] = hdr.attrs[key]
        return attrs

    def get_high_res_dm_mass(self, run, snap):
        hdr = self.get_header(run, snap)
        return hdr['MassTable'][1]
    
    def get_contamination_dm(self, group_catalog):
        keys = group_catalog.keys()
        try:
            assert("GroupMassType" in keys)
        except AssertionError:
            raise(AssertionError("Need to Load in GroupMassType into group catalog"))

        masses = group_catalog['GroupMassType']
        return masses[:, 2] / np.sum(masses, axis=1)
        
    def get_target_fof_index(self, run, snap, target_mass, max_dm=0.25, max_contam=0.25, DMO=False):
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

    def get_sublink_mpb(self, run, snap, subhalo_idx=-1, DMO=False):
        sublink_tree = self.read_sublink_cat(run, DMO=DMO)    
        
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

        fpID = sublink_tree['FirstProgenitorID'][target] ## get target's first progenitor
        while fpID != -1: ## add progenitor info
            for key in sublink_tree.keys():
                cat[key] += [sublink_tree[key][fpID]]
            fpID = sublink_tree['FirstProgenitorID'][fpID]

        cat = {
            key: np.array(cat[key]) for key in cat
        }
            
        return cat

    def get_sublink_tree(self, run, snap, subhalo_idx=-1, DMO=False):
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
        
    def get_target_central_subhalo_index(self, run, snap, target_mass, max_dm=0.25, max_contam=0.25):
        h = self.get_h(run, snap)
        grp_cat = self.read_group_catalog(run, snap, keys=['GroupFirstSub'])
        fof_idx = self.get_target_fof_index(run, snap, target_mass, max_dm=max_dm, max_contam=max_contam)

        return grp_cat['GroupFirstSub'][fof_idx]
    
    def get_contamination_baryon(self, run, snap, fof_idx=-1, subhalo_idx=-1):
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

    def load_single_halo(self, run, snap, fof_idx=-1, part_types=[], keys=[], DMO=False):
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
    
        path_func = self.dir if not DMO else self.dir_dmo
        path = f"{path_func('Sims')}/{self.box_or_run}_{run}/snap_{snap:03d}.hdf5"
        self.check_path(path, 'Snapshot', run, snap)
    
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

    def match_halo_hydro_dmo(self, run, snap, target_mass,
                             mass_tolerance=0.5, contamination_tolerance=0.2,
                             full_search=True):
        '''
        Full search = True compares against every single halo in the box regardless of mass or contamination
        Full search = False assumes that your dmo target is within mass_tolerance and contamination_tolerance
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
        if not full_search:
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

    def load_consistent_trees():
        print('TODO')
        return

    def load_rockstar():
        print('TODO')
        return
        
if __name__ == "__main__":
    print('Hello World!')