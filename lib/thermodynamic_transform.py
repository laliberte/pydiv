"""
Frederic Laliberte 12/2013
"""
import sys
import numpy as np
from netCDF4 import Dataset, num2date
import datetime as dt
import copy
import pkg_resources

def heat_engine_science(options):
    #LOAD THE DATA:
    data = Dataset(options.in_netcdf_file,'r')

    output = Dataset(options.out_netcdf_file,'w',format='NETCDF4')
    output = replicate_netcdf_file(output,data)

    #CONVERT THE DIST VARS TO BIN VALUES AND PUT THEM ALONG THE SAME AXIS
    lengths = [150,75,105]
    #lengths = [150,75]
    #lengths = [150,]
    #lengths = [10,9]
    #COMPUTE THE SIZE OF THE PHASE SPACE:
    phase_space_length=compute_phase_space_length(lengths)

    dist_def={ var:{'vars':['s_moist','hus','pa'],'lengths':lengths,'phase_space_length':phase_space_length} for var in ['dmass','mass','ua','va','wa']}
    #dist_def={ var:{'vars':['s_moist','hus'],'lengths':lengths,'phase_space_length':phase_space_length} for var in ['dmass','mass','ua','va','wa']}
    #dist_def={ var:{'vars':['s_moist',],'lengths':lengths,'phase_space_length':phase_space_length} for var in ['dmass','mass','ua','va','wa']}

    dist_def['mass']['dims'] = ['lat','lon','lev']
    dist_def['dmass']['dims'] = ['lat','lon','lev']
    dist_def['ua']['dims'] = ['lat','slon','lev']
    dist_def['va']['dims'] = ['slat','lon','lev']
    dist_def['wa']['dims'] = ['lat','lon','slev']

    var_list=['mass','ua','va','wa']
    for var in var_list:
        jd_space=create_fluxes_science(data,var,dist_def)
        fluxes = np.apply_along_axis(compute_vector_jd_got,-1,
                                         jd_space,
                                         dist_def[var])
        output_conversion_fluxes(data,output,var,fluxes,options,dist_def[var])
        if var=='mass':
            for type in ['previous','next']:
                jd_space=create_fluxes_science(data,var,dist_def,mass=type)
                mass = np.apply_along_axis(compute_vector_jd,-1,
                                                 jd_space,
                                                 dist_def[var])
                output_conversion_mass(data,output,var,mass,options,dist_def[var],type)
        output.sync()

    if options.divergence:
        for var in ['dmass']:
            output_conversion_fluxes(data,output,var,fluxes,options,dist_def[var])
            for type in ['centre']:
                jd_space=create_fluxes_science(data,var,dist_def,mass=type)
                dmass = np.apply_along_axis(compute_vector_jd,-1,
                                                 jd_space,
                                                 dist_def[var])
                output_conversion_mass(data,output,var,dmass,options,dist_def[var],type)
        output.sync()

        output_fluxes=output.groups['mass_fluxes']
        for gotvar_id,gotvar in enumerate(output_fluxes.groups.keys()):
            output_grp=output_fluxes.groups[gotvar]
            if gotvar_id==0:
                temp=np.zeros(output_grp.variables['va_diff'].shape)
                dims=output_grp.variables['va_diff'].dimensions
            for var in var_list:
                if var=='mass':
                    temp[1:-1,...]+=output_grp.variables[var+'_diff'][:-1,...]
                else:
                    temp+=output_grp.variables[var+'_diff'][:]

        output_fluxes.createVariable('divergence','d',dims)
        output_fluxes.variables['divergence'][:]=temp
        output.sync()
    output.close()           
    return

class discretizations_science():
    def __init__(self,dist_def):
        self.cp=1001
        self.Lv=2.5e6
        self.T0=273.15
        self.dist_def=dist_def
        self.disc_dict={
                's_moist':{'min':220.0,'delta':1.0},
                'hus':{'min':0.0,'delta':1.0},
                'pa':{'min':10.0e2,'delta':10.0e2}}

    def conversion(self,var):
        if var=='s_moist':
            return (lambda x: self.T0*(x/self.cp+1))
        elif var=='hus':
            return (lambda x: self.Lv/self.cp*x)
        elif var=='pa':
            return (lambda x: x)

    def inv_conversion(self,var):
        if var=='s_moist':
            return (lambda x: (x/self.T0-1)*self.cp)
        elif var=='hus':
            return (lambda x: x/(self.Lv/self.cp))
        elif var=='pa':
            return (lambda x: x)

    def discretization(self,var):
        return (lambda x: discretize_max((x-self.disc_dict[var]['min'])/self.disc_dict[var]['delta'],
                                                    self.dist_def['lengths'][self.dist_def['vars'].index(var)])+0.5)

    def inv_discretization(self,var):
        return (lambda y: y*self.disc_dict[var]['delta']+self.disc_dict[var]['min'])

def output_conversion_fluxes(data,output,var,fluxes,options,dist_def):
    disc=discretizations_science(dist_def)

    if not 'mass_fluxes' in output.groups.keys():
        output.createGroup('mass_fluxes')
    output_fluxes=output.groups['mass_fluxes']

    flux_var='flux_'+var
    if flux_var in data.groups.keys():
        data_grp=data.groups[flux_var]
    else:
        data_grp=data.groups[var]
    for time_type in ['time','stime']:
        if time_type in data_grp.dimensions.keys():
            time_var=time_type

    if not time_var in output_fluxes.dimensions.keys():
        output_fluxes.createDimension(time_var,len(data_grp.variables[time_var]))
        output_fluxes.createVariable(time_var,'d',(time_var,))
        output_fluxes.variables[time_var][:]=data_grp.variables[time_var][:]
        output_fluxes.variables[time_var].setncattr('units',data_grp.variables[time_var].units)
        output_fluxes.variables[time_var].setncattr('calendar',data_grp.variables[time_var].calendar)

    for gotvar in fluxes.dtype.names:
        if not 's'+gotvar.capitalize() in output.dimensions.keys():
            length=dist_def['lengths'][dist_def['vars'].index(gotvar)]
            output.createDimension('s'+gotvar.capitalize(),length-1)
            output.createVariable('s'+gotvar.capitalize(),'d',('s'+gotvar.capitalize(),))
            output.variables['s'+gotvar.capitalize()][:]=disc.inv_conversion(gotvar)(disc.inv_discretization(gotvar)(np.arange(1,length)))
        for other_gotvar in fluxes.dtype.names:
            if (not other_gotvar.capitalize() in output.dimensions.keys()):
                length=dist_def['lengths'][dist_def['vars'].index(other_gotvar)]
                output.createDimension(other_gotvar.capitalize(),length)
                output.createVariable(other_gotvar.capitalize(),'d',(other_gotvar.capitalize(),))

                output.variables[other_gotvar.capitalize()][:]=disc.inv_conversion(other_gotvar)(disc.inv_discretization(other_gotvar)(np.arange(0,length)+0.5))

        if not gotvar in output_fluxes.groups.keys():
            output_fluxes.createGroup(gotvar)
        output_grp=output_fluxes.groups[gotvar]
        
        phase_space_dims=tuple(['s'+ps_dim.capitalize() if ps_dim==gotvar else ps_dim.capitalize() for ps_dim in dist_def['vars']])
        output_grp.createVariable(var,'d',(time_var,)+phase_space_dims,zlib=options.compression)

        #if len(fluxes[gotvar].shape)>2:
        #    shape=list(fluxes[gotvar].shape[:-1]+output_grp.variables[var].shape[-len(dist_def['vars']):])
        #    index_staggered = list(output_grp.variables[var].dimensions).index('s'+gotvar.capitalize())+len(fluxes[gotvar].shape)-2
        #    shape[index_staggered]+=1
        #
        #    temp=np.sum(np.take(
        #                np.reshape(fluxes[gotvar],tuple(shape),order='F'),
        #                    range(1,shape[index_staggered]),axis=index_staggered),axis=tuple(range(1,len(fluxes[gotvar].shape)-1)))
        #else:
        shape=list(output_grp.variables[var].shape)
        index_staggered = list(output_grp.variables[var].dimensions).index('s'+gotvar.capitalize())
        shape[index_staggered]+=1
        temp=np.take(
                    np.reshape(fluxes[gotvar],tuple(shape),order='F'),
                        range(1,shape[index_staggered]),axis=index_staggered)
        output_grp.variables[var][:]=temp

        if options.divergence:
            phase_space_dims=tuple([ps_dim.capitalize() for ps_dim in dist_def['vars']])
            output_grp.createVariable(var+'_diff','d',(time_var,)+phase_space_dims)

            shape_cat = copy.copy(shape)
            shape_cat[index_staggered]=1

            #if len(fluxes[gotvar].shape)>2:
            #    shape=list(fluxes[gotvar].shape[:-1]+output_grp.variables[var+'_diff'].shape[-len(dist_def['vars']):])
            #    index_staggered = list(output_grp.variables[var+'_diff'].dimensions).index(gotvar.capitalize())+len(fluxes[gotvar].shape)-2
            #    #shape[index_staggered]+=1
            #    shape_cat = copy.copy(shape)
            #    shape_cat[index_staggered]=1
            #    output_grp.variables[var+'_diff'][:]=np.sum(np.diff(
            #                np.concatenate(
            #                    (
            #                    np.reshape(fluxes[gotvar],tuple(shape),order='F'),
            #                    np.zeros(shape_cat),
            #                    ),
            #                    axis=index_staggered
            #                    ),axis=index_staggered),axis=tuple(range(1,len(fluxes[gotvar].shape)-1)))
            #else:
            shape=list(output_grp.variables[var+'_diff'].shape)
            index_staggered = list(output_grp.variables[var+'_diff'].dimensions).index(gotvar.capitalize())
            shape_cat = copy.copy(shape)
            shape_cat[index_staggered]=1
            output_grp.variables[var+'_diff'][:]=np.diff(
                            np.concatenate(
                            (
                            np.reshape(fluxes[gotvar],tuple(shape),order='F'),
                            np.zeros(shape_cat),
                            ),
                            axis=index_staggered
                            ),axis=index_staggered)
    return

def output_conversion_mass(data,output,var,mass,options,dist_def,type):
    if not 'mass' in output.groups.keys():
        output.createGroup('mass')
    output_mass=output.groups['mass']

    flux_var='flux_'+var
    if flux_var in data.groups.keys():
        data_grp=data.groups[flux_var]
    else:
        data_grp=data.groups[var]
    for time_type in ['time','stime']:
        if time_type in data_grp.dimensions.keys():
            time_var=time_type

    if not time_var in output_mass.dimensions.keys():
        output_mass.createDimension(time_var,len(data_grp.variables[time_var]))
        output_mass.createVariable(time_var,'d',(time_var,))
        output_mass.variables[time_var][:]=data_grp.variables[time_var][:]
        output_mass.variables[time_var].setncattr('units',data_grp.variables[time_var].units)
        output_mass.variables[time_var].setncattr('calendar',data_grp.variables[time_var].calendar)

    phase_space_dims=tuple([ps_dim.capitalize() for ps_dim in dist_def['vars']])
    output_mass.createVariable(var+'_'+type,'d',(time_var,)+phase_space_dims,zlib=options.compression)

    shape=list(mass.shape[:-1]+output_mass.variables[var+'_'+type].shape[-len(dist_def['vars']):])

    output_mass.variables[var+'_'+type][:]=np.sum(
                                    np.reshape(mass,tuple(shape),order='F'),axis=tuple(range(1,len(mass.shape)-1)))

    phase_space_dims=tuple([ps_dim.capitalize() for ps_dim in dist_def['vars']])
    output_mass.createVariable(var+'_'+type+'_diff','d',(time_var,)+phase_space_dims)
    shape=list(mass.shape[:-1]+output_mass.variables[var+'_'+type].shape[-len(dist_def['vars']):])
    shape[0]-=1

    output_mass.variables[var+'_'+type+'_diff'][:-1,...]=np.sum(
                                        np.reshape(
                                        mass[1:,...]-mass[:-1,...]
                                        ,tuple(shape),order='F')
                                        ,axis=tuple(range(1,len(mass.shape)-1)))
    output_mass.variables[var+'_'+type+'_diff'][-1,...]=0.0
    return

def discretize_max(x,max):
    disc=np.floor(x)
    disc[disc<=0]=0
    disc[disc>=max-1]=max-1
    return disc

def add_dimension(data,output,dist_def,var):
   #FIND WHICH DIMENSIONS TO INCLUDE
   var_dims = list(data.variables[var].dimensions)
   for dims in dist_def['dims']: 
       if dims != 'time': var_dims.remove(dims)
   var_dims.append(bins_name)

   #CHECK IF THE NEW VAR DIMENSIONS EXIST. IF NOT, ADD THEM
   for dims in var_dims:
     if dims not in output.dimensions.keys():
        if dims not in dist_def['dims']: #MEANS THAT IT IS NOT TIME
           output.createDimension(dims,len(data.dimensions[dims]))
           dim_var = output.createVariable(dims,'d',(dims,))
           dim_var[:] = data.variables[dims][:]
           output = replicate_netcdf_var(output,data,dims)
        else: #USE THE FIRST TIME STEP
           output.createDimension(dims,1) #MAKE TIME RECORD DIMENSION
           dim_var = output.createVariable(dims,'d',(dims,))
           dim_var[:] = data.variables[dims][0]
           output = replicate_netcdf_var(output,data,dims)
   return output, var_dims

def compute_phase_space_length(bins_length):
       return int(np.cumprod(bins_length).sum() - 1 - np.cumprod(bins_length[:-1]).sum())

def create_structured_array(data,var,dist_def):
       #CREATE A STRUCTURED ARRAY:
       list_types=[('flux',np.float)]
       for gotvar in dist_def['vars']:
            list_types.append((gotvar,np.float))
            list_types.append((gotvar+'_MASK',np.int8))
            list_types.append((gotvar+'_TOTAL',np.int32))
            list_types.append((gotvar+'_SUM',np.float))
       return list_types

def create_fluxes_science(data,var,dist_def,mass=None):
    disc=discretizations_science(dist_def[var])

    if var=='dmass':
        flux_var=var
    else:
        flux_var='flux_'+var
    list_types=create_structured_array(data.groups[flux_var],flux_var,dist_def[var])
    fluxes =collapse_dims_last(data.groups[flux_var],flux_var,dist_def[var]['dims'])

    #if var=='dmass':
    #    fluxes=np.zeros(fluxes.shape)
    #    for flux_var in ['mass','ua','va','wa']:
    #        if flux_var == 'mass':
    #            fluxes+=collapse_dims_last_var(
    #                                diff_with_homogeneous_bc(data.groups['flux_'+flux_var].variables['flux_'+flux_var][:],axis=0),
    #                                data.groups['flux_'+flux_var].variables['flux_'+flux_var].dimensions ,dist_def[flux_var]['dims'])
    #        elif flux_var == 'wa':
    #            fluxes+=collapse_dims_last_var(
    #                                diff_with_homogeneous_bc(data.groups['flux_'+flux_var].variables['flux_'+flux_var][:],axis=1),
    #                                data.groups['flux_'+flux_var].variables['flux_'+flux_var].dimensions ,dist_def[flux_var]['dims'])
    #        elif flux_var == 'va':
    #            fluxes+=collapse_dims_last_var(
    #                                diff_with_homogeneous_bc(data.groups['flux_'+flux_var].variables['flux_'+flux_var][:],axis=2),
    #                                data.groups['flux_'+flux_var].variables['flux_'+flux_var].dimensions ,dist_def[flux_var]['dims'])
    #        elif flux_var == 'ua':
    #            fluxes+=collapse_dims_last_var(
    #                                np.roll(data.groups['flux_'+flux_var].variables['flux_'+flux_var][:],-1,axis=-1)-
    #                                data.groups['flux_'+flux_var].variables['flux_'+flux_var][:],
    #                                data.groups['flux_'+flux_var].variables['flux_'+flux_var].dimensions ,dist_def[flux_var]['dims'])

    jd_space=np.zeros(fluxes.shape,dtype=list_types)
    jd_space['flux']=fluxes

    dimensions=[dim[1:] if dim[0]=='s' else dim for dim in dist_def[var]['dims'] ]
    #dimensions=dist_def['mass']['dims']

    if mass=='previous':
        for gotvar in dist_def[var]['vars']:
            jd_space[gotvar]=disc.discretization(gotvar)(disc.conversion(gotvar)(
                                collapse_dims_last_var(data.groups[gotvar].variables[gotvar][:-1,...],
                                                     data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
                                        ))
        return jd_space
    elif mass=='next':
        for gotvar in dist_def[var]['vars']:
            jd_space[gotvar]=disc.discretization(gotvar)(disc.conversion(gotvar)(
                                collapse_dims_last_var(data.groups[gotvar].variables[gotvar][1:,...],
                                                     data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
                                        ))
        return jd_space
    elif mass=='centre':
        for gotvar in dist_def[var]['vars']:
            jd_space[gotvar]=disc.discretization(gotvar)(disc.conversion(gotvar)(
                                collapse_dims_last_var(data.groups[gotvar].variables[gotvar][:,...],
                                                     data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
                                        ))
        return jd_space




    for gotvar in dist_def[var]['vars']:
        if var == 'mass':
            gotvar_n=collapse_dims_last_var(data.groups[gotvar].variables[gotvar][1:,...],
                                                data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
            gotvar_p=collapse_dims_last_var(data.groups[gotvar].variables[gotvar][:-1,...],
                                                data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
        elif var == 'wa':
            gotvar_n=collapse_dims_last_var(data.groups[gotvar].variables[gotvar][:,1:,:,:],
                                                data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
            gotvar_p=collapse_dims_last_var(data.groups[gotvar].variables[gotvar][:,:-1,:,:],
                                                data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
        elif var == 'va':
            gotvar_n=collapse_dims_last_var(data.groups[gotvar].variables[gotvar][:,:,1:,:],
                                                data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
            gotvar_p=collapse_dims_last_var(data.groups[gotvar].variables[gotvar][:,:,:-1,:],
                                                data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
        elif var == 'ua':
            gotvar_n=collapse_dims_last_var(data.groups[gotvar].variables[gotvar][...],
                                                data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
            gotvar_p=collapse_dims_last_var(np.roll(data.groups[gotvar].variables[gotvar][...],1,axis=-1),
                                                data.groups[gotvar].variables[gotvar].dimensions ,dimensions)

        gotvar_p=disc.discretization(gotvar)(disc.conversion(gotvar)(gotvar_p))
        gotvar_n=disc.discretization(gotvar)(disc.conversion(gotvar)(gotvar_n))

        jd_space[gotvar]=gotvar_p
        jd_space[gotvar+'_MASK']=np.sign(gotvar_n-gotvar_p)
        jd_space[gotvar+'_TOTAL']=gotvar_n-gotvar_p
        jd_space[gotvar+'_SUM']=np.zeros(jd_space[gotvar+'_MASK'].shape)
    return jd_space

def diff_with_homogeneous_bc(array,axis=-1):
    return np.concatenate((np.take(array,[0],axis=axis),
                   np.diff(array,axis=axis),
                   -np.take(array,[-1],axis=axis)),axis=axis)

def got_digitize_floor(array,length):
    out=np.rint(np.floor(array))
    out[out<0]=0
    out[out>length-1]=length-1
    #out[out>length]=length
    return out

def compute_vector_jd_got(jd_space,dist_def):
    #CREATE OUPTUT:
    list_types=[(gotvar,np.float) for gotvar in dist_def['vars']]
    jd_out=np.zeros((dist_def['phase_space_length']+1,),list_types)

    while max([np.abs(jd_space[gotvar+'_TOTAL']).max() for gotvar in dist_def['vars']])>0:
        for gotvar in dist_def['vars']:
            binning_mask = create_binning_mask(dist_def,jd_space,gotvar)
            if np.any(binning_mask):
                jd_space, jd_out = bin_gotvar_over_mask(dist_def,jd_space,jd_out,gotvar,binning_mask)
    return jd_out

def create_binning_mask(dist_def,jd_space,gotvar):
    #SIMPLE UPWIND:
    #binning_mask=(np.abs(jd_space[gotvar+'_MASK'])>0)
    #MID_POINT:
    binning_mask=(np.abs(jd_space[gotvar+'_MASK'])>0)
    fix_mask=(np.abs(jd_space[gotvar+'_MASK'])>0)
    for dist_var in [var for var in dist_def['vars'] if var!=gotvar]:
        valid_mask=np.logical_and(fix_mask,(np.abs(jd_space[dist_var+'_MASK'])>0))
        if np.any(valid_mask):
            binning_mask[valid_mask]&=np.less_equal(ratio_next(jd_space,gotvar,valid_mask),
                                               ratio_next(jd_space,dist_var,valid_mask))
    return binning_mask

def ratio_next(jd_space,var,mask):
    return ((jd_space[var+'_SUM'][mask]+0.5*jd_space[var+'_MASK'][mask])/
            (jd_space[var+'_SUM'][mask]+1.0*jd_space[var+'_TOTAL'][mask]))

def bin_gotvar_over_mask(dist_def,jd_space,jd_out,gotvar,binning_mask):
    #Applies the transform:
    #CREATE BIN ARRAY
    jd_space[gotvar+'_SUM'][binning_mask]+=0.5*jd_space[gotvar+'_MASK'][binning_mask]
    #print jd_space.dtype,jd_space
    bin_array = np.zeros_like(jd_space['flux'][binning_mask], dtype=int)
    for nv,dvar in enumerate(dist_def['vars']):
        bin_array+=np.prod(dist_def['lengths'][0:nv])*\
                           got_digitize_floor(jd_space[dvar][binning_mask]+jd_space[dvar+'_SUM'][binning_mask],dist_def['lengths'][nv])
    if dist_def['phase_space_length'] < bin_array.max(): raise IOError('number of actual bins is larger than expected output')

    #SETUP THE INVERSION
    temp_data = jd_space['flux'][binning_mask]*jd_space[gotvar+'_MASK'][binning_mask]

    #DO THE COMPUTATION USING NP.BINCOUNT
    jd_out[gotvar] += np.bincount(bin_array,
                                 weights=temp_data,
                                 minlength=dist_def['phase_space_length']+1)

    #APPLY THE GRADIENT:
    jd_space[gotvar+'_SUM'][binning_mask]+=0.5*jd_space[gotvar+'_MASK'][binning_mask]
    jd_space[gotvar+'_TOTAL'][binning_mask]-=jd_space[gotvar+'_MASK'][binning_mask]
    jd_space[gotvar+'_MASK'][binning_mask]*=np.where(np.abs(jd_space[gotvar+'_TOTAL'][binning_mask])>0,1,0)
    return jd_space,jd_out

def compute_vector_jd(jd_space,dist_def):
    jd_out = bin_var(dist_def,jd_space)
    return jd_out

def bin_var(dist_def,jd_space):
    #Applies the transform:
    #CREATE BIN ARRAY
    bin_array = np.zeros_like(jd_space['flux'], dtype=int)
    for nv,dvar in enumerate(dist_def['vars']):
        bin_array+=np.prod(dist_def['lengths'][0:nv])*\
                           got_digitize_floor(jd_space[dvar],dist_def['lengths'][nv])
    if dist_def['phase_space_length'] < bin_array.max(): raise IOError('number of actual bins is larger than expected output')

    #SETUP THE INVERSION
    temp_data = jd_space['flux']

    #DO THE COMPUTATION USING NP.BINCOUNT
    jd_out = np.bincount(bin_array,
                                 weights=temp_data,
                                 minlength=dist_def['phase_space_length']+1)

    return jd_out

def collapse_dims_last(data,var,dist_dims,time_id=None):
        #RETRIEVE ONE OF THE VARIABLES
        add_dim_before=(lambda x: np.reshape(x,(1,)+x.shape))
        if time_id!=None:
            temp = add_dim_before(data.variables[var][time_id,...])
        else:
            temp = data.variables[var][:]

	temp_dims = data.variables[var].dimensions
        return collapse_dims_last_var(temp,temp_dims,dist_dims)

def collapse_dims_last_var(temp,temp_dims,dist_dims):
        #FIND THE PERMUTATION VECTOR TO PUT dist_dims IN THE LAST DIMS
	dist_dims_ind = [ temp_dims.index(dims) for dims in dist_dims ]
	other_dims_ind = list(set(range(len(temp_dims))).difference(set(dist_dims_ind)))
        #FIND THE SHAPE VECTOR TO COLLAPSE dist_dims ONTO LAST DIM
	#collapse_shape = [np.prod([temp.shape[dims] for dims in other_dims_ind])] + [np.prod([temp.shape[dims] for dims in dist_dims_ind ])]
	collapse_shape = [temp.shape[dims] for dims in	other_dims_ind] + [np.prod([temp.shape[dims] for dims in dist_dims_ind ])]
        #TRANSPOSE temp TO PUT dist_dims AT THE END
	temp = np.transpose(temp,axes=other_dims_ind+dist_dims_ind)
        #COLLAPSE dist_dims
	temp = np.reshape(temp,collapse_shape)
	return temp

def replicate_netcdf_file(output,data):
	for att in data.ncattrs():
            att_val=getattr(data,att)
            if 'encode' in dir(att_val):
                att_val=att_val.encode('ascii','replace')
	    setattr(output,att,att_val)
        output.history+='\n' 
        #output.history+=dt.datetime.now().strftime('%Y-%m-%d %H:%M') #Add time
        output.history+='joint_distribution '+pkg_resources.get_distribution('pydiv').version
	return output

def replicate_netcdf_var(output,data,var):
	for att in data.variables[var].ncattrs():
	    if att[0]!='_':
                att_val=getattr(data.variables[var],att)
                if 'encode' in dir(att_val):
                    att_val=att_val.encode('ascii','replace')
	        setattr(output.variables[var],att,att_val)
	return output

def replicate_netcdf_var2(output,data,var,var_out):
	for att in data.variables[var].ncattrs():
	    if att[0]!='_':
                att_val=getattr(data.variables[var],att)
                if 'encode' in dir(att_val):
                    att_val=att_val.encode('ascii','replace')
	        setattr(output.variables[var_out],att,att_val)
	return output

def generate_test(options):
    output = Dataset(options.out_netcdf_file,'w',format='NETCDF4')
    dimensions={
               'time':[0,0.5,1.0],
               'stime':[0.25,0.75],
               'lev':[0.75,0.25],
               'slev':[0.5,],
               'lat':[-45.0,45.0],
               'slat':[0.0],
               'lon':[0.0,180.0],
               'slon':[-90.0,90.0]}

    var_list={
              'flux_ua': ('time','lev','lat','slon'),
              'flux_va': ('time','lev','slat','lon'),
              'flux_wa': ('time','slev','lat','lon'),
              'flux_mass': ('stime','lev','lat','lon'),
              's_moist': ('time','lev','lat','lon'),
              'pa': ('time','lev','lat','lon'),
              'hus': ('time','lev','lat','lon'),
              'dmass':('time','lev','lat','lon'),
              'divergence':('time','lev','lat','lon'),
              'mass_diff':('time','lev','lat','lon')}
    for var in var_list.keys():
        output.createGroup(var)
        for dim in var_list[var]:
            output.groups[var].createDimension(dim,len(dimensions[dim]))
            output.groups[var].createVariable(dim,'d',(dim,))
            output.groups[var].variables[dim][:]=dimensions[dim]
            output.groups[var].variables[var_list[var][0]].setncattr('units','days since 1850-01-01 00:00:00')
            output.groups[var].variables[var_list[var][0]].setncattr('calendar','365_day')
        temp=output.groups[var].createVariable(var,'d',var_list[var])
        temp[:]=0.0
            


    var_data={
              'flux_ua':0.5,
              'flux_va':0.5,
              'flux_wa':0.5,
              'flux_mass':1.0
              }

    for var in var_data.keys():
        output.groups[var].variables[var][:]=var_data[var]

    direction='wa'
    if direction=='va':
        var='flux_mass'
        output.groups[var].variables[var][1,:,0,:]=0.5
        output.groups[var].variables[var][1,:,1,:]=1.5
        var='flux_va'
        output.groups[var].variables[var][0,...]=0.0
        output.groups[var].variables[var][-1,...]=0.0
        for var in ['ua','wa']:
            output.groups['flux_'+var].variables['flux_'+var][...]=0.0
    elif direction=='ua':
        var='flux_mass'
        output.groups[var].variables[var][1,:,:,0]=0.5
        output.groups[var].variables[var][1,:,:,1]=1.5
        var='flux_ua'
        output.groups[var].variables[var][0,...]=0.0
        output.groups[var].variables[var][-1,...]=0.0
        output.groups[var].variables[var][:,:,:,0]=0.0
        for var in ['va','wa']:
            output.groups['flux_'+var].variables['flux_'+var][...]=0.0
    elif direction=='wa':
        var='flux_mass'
        output.groups[var].variables[var][1,0,:,:]=0.5
        output.groups[var].variables[var][1,1,:,:]=1.5
        var='flux_wa'
        output.groups[var].variables[var][0,...]=0.0
        output.groups[var].variables[var][-1,...]=0.0
        for var in ['va','ua']:
            output.groups['flux_'+var].variables['flux_'+var][...]=0.0

    var='s_moist'
    output.groups[var].variables[var][:]=((220+np.random.rand(*output.groups[var].variables[var].shape)*9)/273.15-1.0)*1001

    var='hus'
    output.groups[var].variables[var][:]=np.random.rand(*output.groups[var].variables[var].shape)*4*1001/2.5e6

    var='pa'
    output.groups[var].variables[var][:]=100e2+np.random.rand(*output.groups[var].variables[var].shape)*900e2

    for var in ['s_moist','hus','pa']:
        test=copy.copy(output.groups[var].variables[var][0,0,0,0])
        test2=copy.copy(output.groups[var].variables[var][1,0,0,0])
        output.groups[var].variables[var][:]=test
        output.groups[var].variables[var][1:,0,0,0]=test2

    #Divergence:
    var='flux_ua'
    divergence=(np.roll(output.groups[var].variables[var][:],-1,axis=3)-
                   output.groups[var].variables[var][:])
    var='flux_va'
    divergence+=np.concatenate([output.groups[var].variables[var][:],
                                -output.groups[var].variables[var][:]],axis=2)
    var='flux_wa'
    divergence+=np.concatenate([output.groups[var].variables[var][:],
                                -output.groups[var].variables[var][:]],axis=1)

    var='divergence'
    output.groups[var].variables[var][:]=divergence

    var='flux_mass'
    mass_diff=np.diff(output.groups[var].variables[var],axis=0)

    var='mass_diff'
    output.groups[var].variables[var][1:-1,...]=mass_diff

    var='dmass'
    output.groups[var].variables[var][1:-1,...]=mass_diff+divergence[1:-1,...]

    output.setncattr('history','')
    output.close()
    return


def generate_subparser(subparser):
    subparser.add_argument("-z","--zipped",dest="compression",
                      default=False, action="store_true",
                      help="Output file with NetCDF4 compression")
    subparser.add_argument("in_netcdf_file",help="netCDF input file")
    subparser.add_argument("out_netcdf_file",help="netCDF output file")
    return
    

def main():
    import argparse 
    import textwrap

    #Option parser
    version_num='0.6'
    description=textwrap.dedent('''\
    This script computes the thermodynamics transform.
    ''')
    epilog='Version {0}: Frederic Laliberte, Paul Kushner 07/2015\n\
\n\
If using this code please cite:\n\n\
Constrained work output of the moist atmospheric heat engine in a warming climate (2015):\n\
F. Laliberte, J. Zika, L. Mudryk, P. J. Kushner, J. Kjellsson, K. Doos, Science.'.format(version_num)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                            description=description,
                            version='%(prog)s '+version_num,
                            epilog=epilog)

    subparsers = parser.add_subparsers(help='Commands to process the data',dest='command')

    science_parser=subparsers.add_parser('transform_science',
                                           description=textwrap.dedent(
                                                '''Compute the thermodynamics transform of Laliberte et al. (2015).
                                                 '''),epilog=epilog
                                         )
    generate_subparser(science_parser)
    science_parser.add_argument("--divergence",dest="divergence",
                      default=False, action="store_true",
                      help="Compute divergence in phase space")

    test_parser=subparsers.add_parser('transform_test',
                                           description=textwrap.dedent(
                                                '''Compute the thermodynamics transform of Laliberte et al. (2015).
                                                 '''),epilog=epilog
                                         )

    test_parser.add_argument("out_netcdf_file",help="netCDF output file")
    options=parser.parse_args()

    if options.command == "transform_science":
        heat_engine_science(options)
    elif options.command == "transform_test":
        generate_test(options)
    return
      
if __name__ == "__main__":
    main()
