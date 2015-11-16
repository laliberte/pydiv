"""
Frederic Laliberte 09/2015
"""
import sys
import numpy as np
from netCDF4 import Dataset, num2date
import datetime as dt
import copy
import pkg_resources

def coarse_grain(options):
    #CONVERT THE DIST VARS TO BIN VALUES AND PUT THEM ALONG THE SAME AXIS
    lengths = [14,36]
    #COMPUTE THE SIZE OF THE PHASE SPACE:
    phase_space_length=compute_phase_space_length(lengths)

    dist_def={ var:{'vars':['pa','latitude'],
                'lengths':lengths,
                'discretization_type':'coarse',
                'phase_space_length':phase_space_length} for var in ['dmass','mass','ua','va','wa']}

    dist_def['dmass']['dims'] = ['lat','lon','lev']
    dist_def['mass']['dims'] = ['lat','lon','lev']
    dist_def['ua']['dims'] = ['lat','slon','lev']
    dist_def['va']['dims'] = ['slat','lon','lev']
    dist_def['wa']['dims'] = ['lat','lon','slev']

    dist_def['dmass']['time_type'] = 'centre'
    dist_def['mass']['time_type'] = 'previous'
    dist_def['ua']['time_type'] = 'centre'
    dist_def['va']['time_type'] = 'centre' 
    dist_def['wa']['time_type'] = 'centre'

    dist_def['mass']['axis'] = 0
    dist_def['wa']['axis'] = 1
    dist_def['va']['axis'] = 2
    dist_def['ua']['axis'] = 3

    dist_def['mass']['bc'] = 'staggered'
    dist_def['dmass']['bc'] = 'constant'
    dist_def['va']['bc'] = 'staggered'
    dist_def['ua']['bc'] = 'periodic'
    dist_def['wa']['bc'] = 'staggered'

    return perform_transforms(options,dist_def,out_name='coarse_')

def heat_engine_science(options):
    #CONVERT THE DIST VARS TO BIN VALUES AND PUT THEM ALONG THE SAME AXIS
    lengths = [12,125,60,48]
    #COMPUTE THE SIZE OF THE PHASE SPACE:
    phase_space_length=compute_phase_space_length(lengths)

    dist_def={ var:{'vars':['latitude','smoist','hus','pa'],
                'lengths':lengths,
                'discretization_type':'science',
                'phase_space_length':phase_space_length} for var in ['dmass','mass','ua','va','wa']}

    dist_def['dmass']['dims'] = ['lat','lon','lev']
    dist_def['mass']['dims'] = ['lat','lon','lev']
    dist_def['ua']['dims'] = ['lat','slon','lev']
    dist_def['va']['dims'] = ['slat','lon','lev']
    dist_def['wa']['dims'] = ['lat','lon','slev']

    dist_def['dmass']['time_type'] = 'centre'
    dist_def['mass']['time_type'] = 'previous'
    dist_def['ua']['time_type'] = 'centre'
    dist_def['va']['time_type'] = 'centre' 
    dist_def['wa']['time_type'] = 'centre'

    dist_def['mass']['axis'] = 0
    dist_def['wa']['axis'] = 1
    dist_def['va']['axis'] = 2
    dist_def['ua']['axis'] = 3

    dist_def['mass']['bc'] = 'staggered'
    dist_def['dmass']['bc'] = 'constant'
    dist_def['va']['bc'] = 'staggered'
    dist_def['ua']['bc'] = 'periodic'
    dist_def['wa']['bc'] = 'staggered'

    return perform_transforms(options,dist_def,out_name='tt_')

def projection_science(options):
    #CONVERT THE DIST VARS TO BIN VALUES AND PUT THEM ALONG THE SAME AXIS
    lengths_dict = { 'smoist,ta': [12,125,120],
                     'hus,gh_gc': [12,60,120],
                     'hus,gc_gv': [12,60,120],
                     'hus,gd': [12,60,120],
                     'pa,thetav': [12,48,125]}
    lengths=lengths_dict[options.coordinates]
    #COMPUTE THE SIZE OF THE PHASE SPACE:
    phase_space_length=compute_phase_space_length(lengths)

    dist_def={ var:{'vars':['latitude',]+options.coordinates.split(','),
                'lengths':lengths,
                'discretization_type':'science',
                'phase_space_length':phase_space_length} for var in ['dmass','mass','latitude','smoist','pa','hus']}

    dist_def['dmass']['dims'] = ['Latitude','Smoist','Hus','Pa']
    dist_def['mass']['dims'] = ['Latitude','Smoist','Hus','Pa']
    dist_def['latitude']['dims'] = ['sLatitude','Smoist','Hus','Pa']
    dist_def['smoist']['dims'] = ['Latitude','sSmoist','Hus','Pa']
    dist_def['hus']['dims'] = ['Latitude','Smoist','sHus','Pa']
    dist_def['pa']['dims'] = ['Latitude','Smoist','Hus','sPa']

    dist_def['dmass']['time_type'] = 'full'
    dist_def['mass']['time_type'] = 'full'
    dist_def['latitude']['time_type'] = 'full'
    dist_def['smoist']['time_type'] = 'full'
    dist_def['hus']['time_type'] = 'full' 
    dist_def['pa']['time_type'] = 'full'

    dist_def['mass']['axis'] = 0
    dist_def['latitude']['axis'] = 1
    dist_def['smoist']['axis'] = 2
    dist_def['hus']['axis'] = 3
    dist_def['pa']['axis'] = 4

    dist_def['mass']['bc'] = 'constant'
    dist_def['dmass']['bc'] = 'constant'
    dist_def['latitude']['bc'] = 'staggered'
    dist_def['smoist']['bc'] = 'staggered'
    dist_def['hus']['bc'] = 'staggered'
    dist_def['pa']['bc'] = 'staggered'

    return perform_transforms(options,dist_def,source_name='tt_',out_name=options.coordinates.replace(',','_')+'_')

def perform_transforms(options,dist_def,source_name='',out_name=''):
    #LOAD THE DATA:
    data = Dataset(options.in_netcdf_file,'r')

    output = Dataset(options.out_netcdf_file,'w',format='NETCDF4')
    output = replicate_netcdf_file(output,data)

    var_list=dist_def.keys()
    for var in var_list:
        print var
        if var =='mass':
            group_name='masstendency_'+source_name+var
        elif var=='dmass':
            group_name='sources_'+source_name+var
        else:
            group_name='massfluxes_'+source_name+var
        sub_var_list=[flux_var for flux_var in data.groups[group_name].variables.keys() if
                                set(data.groups[group_name].variables[flux_var].dimensions).issuperset(dist_def[var]['dims'])]
        for flux_var in sub_var_list:
            perform_transform_one_var(data,output,group_name,flux_var,var,dist_def,options,source_name,out_name)
            if 'quantity' in dir(options):
                for quantity in options.quantity:
                    perform_transform_one_var(data,output,group_name,flux_var,var,dist_def,options,source_name,out_name+quantity+'_',quantity=quantity)

    if 'quantity' in dir(options):
        for quantity in options.quantity:
            compute_sum_massfluxes(output,dist_def,out_name+quantity+'_',options)
    compute_sum_massfluxes(output,dist_def,out_name,options)
        
    if options.divergence:
        compute_divergence(output,dist_def,out_name,options)
    output.close()           
    return

def perform_transform_one_var(data,output,group_name,flux_var,var,dist_def,options,source_name,out_name,quantity=''):
    if dist_def[var]['bc'] != 'constant':
        jd_space, checks=create_massfluxes_science(data,group_name,flux_var,var,dist_def,options,quantity=quantity)
        fluxes = np.apply_along_axis(compute_vector_jd_got,-1,
                                         jd_space,
                                         dist_def[var])
        output_conversion_massfluxes(data,output,group_name,flux_var,var,fluxes,options,dist_def[var],source_name,out_name)
        output.sync()
        if not 'weight' in flux_var.split('_'):
            output_conversion_checks(data,output,group_name,flux_var,var,checks,options,source_name,out_name)
        output.sync()

    if var in ['mass']:
        jd_space=create_masstendency_science(data,group_name,flux_var,var,dist_def,quantity=quantity)
        mass = np.apply_along_axis(compute_vector_jd,-1,
                                         jd_space,
                                         dist_def[var])
        output_conversion_masstendency(data,output,group_name,flux_var,var,mass,options,dist_def[var],source_name,out_name)
        output.sync()

    if var in ['dmass']:
        jd_space=create_sources_science(data,group_name,flux_var,var,dist_def,quantity=quantity)
        mass = np.apply_along_axis(compute_vector_jd,-1,
                                         jd_space,
                                         dist_def[var])
        output_conversion_sources(data,output,group_name,flux_var,var,mass,options,dist_def[var],source_name,out_name)
        output.sync()
    return

def compute_sum_massfluxes(output,dist_def,out_name,options):
    var_ref=dist_def.keys()[0]
    for gotvar_id,gotvar in enumerate(dist_def[var_ref]['vars']):
        dims=['s'+var.capitalize() if var==gotvar else var.capitalize() for var in dist_def[var_ref]['vars']]
        out_group_name='massfluxes_'+out_name+gotvar
        output_grp=output.groups[out_group_name]
        excluded_keywords=['sum','weight','check','diff']
        var_list=[var for var in output.groups[out_group_name].variables.keys() if 
                                 ( set(output.groups[out_group_name].variables[var].dimensions).issuperset(dims) and
                                   len(set(var.split('_')).intersection(excluded_keywords))==0 )]
        for var_id,var in enumerate(var_list):
            if var_id==0:
                temp=np.zeros(output_grp.variables[var].shape)
            temp+=output_grp.variables[var][:]
        output_grp.createVariable('massfluxes_sum','f',output_grp.variables[var].dimensions,zlib=options.compression)
        output_grp.variables['massfluxes_sum'][:]=temp
        output_grp.variables['massfluxes_sum'].setncattr('formula','+'.join(var_list))
        output.sync()
    return

def compute_divergence(output,dist_def,out_name,options):
    #Compute divergence in thermal space:
    var_ref=dist_def.keys()[0]
    for gotvar_id,gotvar in enumerate(dist_def[var_ref]['vars']):
        source_group_name='massfluxes_'+out_name+gotvar
        output_grp=output.groups[source_group_name]
        var_list=[var for var in output_grp.variables.keys() if 
                             var.split('_')[-1]=='diff' ]
        if gotvar_id==0:
            temp=np.zeros(output_grp.variables[var_list[0]].shape)
            dims=output_grp.variables[var_list[0]].dimensions
        for var in var_list:
            temp+=output_grp.variables[var][:]

    out_group_name='massfluxes_'+out_name+'divergence' 
    if not out_group_name in output.groups.keys():
        output.createGroup(out_group_name)
    output_div=output.groups[out_group_name]

    time_var='time'
    if not time_var in output_div.dimensions.keys():
        output_div.createDimension(time_var,len(output_grp.variables[time_var]))
        output_div.createVariable(time_var,'d',(time_var,))
        output_div.variables[time_var][:]=output_grp.variables[time_var][:]
        output_div.variables[time_var].setncattr('units',output_grp.variables[time_var].units)
        output_div.variables[time_var].setncattr('calendar',output_grp.variables[time_var].calendar)

    disc=discretizations_class(dist_def[dist_def.keys()[0]])

    for other_gotvar in dist_def[dist_def.keys()[0]]['vars']:
        if (not other_gotvar.capitalize() in output_div.dimensions.keys()):
            length=dist_def[dist_def.keys()[0]]['lengths'][dist_def[dist_def.keys()[0]]['vars'].index(other_gotvar)]
            output_div.createDimension(other_gotvar.capitalize(),length)
            output_div.createVariable(other_gotvar.capitalize(),'d',(other_gotvar.capitalize(),))

            output_div.variables[other_gotvar.capitalize()][:]=disc.inv_conversion(other_gotvar)(disc.inv_discretization(other_gotvar)(np.arange(0,length)+0.5))
    output_div.createVariable('divergence','f',dims,zlib=options.compression)
    output_div.variables['divergence'][:]=temp

    output.sync()
    return

class discretizations_class():
    def __init__(self,dist_def):
        self.cp=1001
        self.Lv=2.5e6
        self.T0=273.15
        self.dist_def=dist_def
        if self.dist_def['discretization_type']=='science':
            self.disc_dict={
                    'latitude':{'min':-90.0,'delta':15.0},
                    'smoist':{'min':245.0,'delta':1.0},
                    'hus':{'min':0.0,'delta':1.0},
                    'pa':{'min':100.0e2,'delta':20.0e2},
                    'ta':{'min':200.0,'delta':1.0},
                    'gh_gc':{'min':-1e5,'delta':5e3},
                    'gc_gv':{'min':-1e5,'delta':5e3},
                    'gd':{'min':-1e5,'delta':5e3},
                    'thetav':{'min':245.0,'delta':1.0}
                    }
        elif self.dist_def['discretization_type']=='coarse':
            self.disc_dict={
                    'latitude':{'min':-90.0,'delta':5.0},
                    'longitude':{'min':-180.0,'delta':10.0},
                    'pa':{'min':0.0,'delta':100.0e2}
                    }

    def conversion(self,var):
        if var=='smoist':
            return (lambda x: self.T0*(x/self.cp+1))
        elif var=='hus':
            return (lambda x: self.Lv/self.cp*x)
        else:
            return (lambda x: x)

    def inv_conversion(self,var):
        if var=='smoist':
            return (lambda x: (x/self.T0-1)*self.cp)
        elif var=='hus':
            return (lambda x: x/(self.Lv/self.cp))
        else:
            return (lambda x: x)

    def discretization(self,var):
        return (lambda x: discretize_max((x-self.disc_dict[var]['min'])/self.disc_dict[var]['delta'],
                                                    self.dist_def['lengths'][self.dist_def['vars'].index(var)])+0.5)

    def inv_discretization(self,var):
        return (lambda y: y*self.disc_dict[var]['delta']+self.disc_dict[var]['min'])

def output_conversion_checks(data,output,source_group_name,flux_var,var,checks,options,source_name,out_name):
    if source_group_name in data.groups.keys():
        data_grp=data.groups[source_group_name]
    else:
        data_grp=data.groups[var]

    for time_type in ['time','stime']:
        if time_type in data_grp.dimensions.keys():
            time_var=time_type
    time_var='time'

    for gotvar in checks.keys():
        out_group_name='massfluxes_'+out_name+gotvar
        if not out_group_name in output.groups.keys():
            output.createGroup(out_group_name)
        output_checks=output.groups[out_group_name]
        #if not 'checks' in output_checks.groups.keys():
        #    output_checks.createGroup('checks')
        #output_checks=output_checks.groups['checks']

        if not time_var in output_checks.dimensions.keys():
            output_checks.createDimension(time_var,len(data_grp.variables[time_var]))
            output_checks.createVariable(time_var,'d',(time_var,))
            output_checks.variables[time_var][:]=data_grp.variables[time_var][:]
            output_checks.variables[time_var].setncattr('units',data_grp.variables[time_var].units)
            output_checks.variables[time_var].setncattr('calendar',data_grp.variables[time_var].calendar)

        if ('check' in dir(options) and options.check!=None): 
            for checkvar_id, checkvar in enumerate(options.check):
                temp=output_checks.createVariable(flux_var+'_check_'+checkvar,'f',(time_var,))
                temp[:]=checks[gotvar][checkvar_id]
    output.sync()
    return

def output_conversion_massfluxes(data,output,source_group_name,flux_var,var,fluxes,options,dist_def,source_name,out_name):
    disc=discretizations_class(dist_def)

    if source_group_name in data.groups.keys():
        data_grp=data.groups[source_group_name]
    else:
        data_grp=data.groups[var]

    for time_type in ['time','stime']:
        if time_type in data_grp.dimensions.keys():
            time_var=time_type
    time_var='time'


    for gotvar in fluxes.dtype.names:
        out_group_name='massfluxes_'+out_name+gotvar
        if not out_group_name in output.groups.keys():
            output.createGroup(out_group_name)
        output_fluxes=output.groups[out_group_name]

        if not time_var in output_fluxes.dimensions.keys():
            output_fluxes.createDimension(time_var,len(data_grp.variables[time_var]))
            output_fluxes.createVariable(time_var,'d',(time_var,))
            output_fluxes.variables[time_var][:]=data_grp.variables[time_var][:]
            output_fluxes.variables[time_var].setncattr('units',data_grp.variables[time_var].units)
            output_fluxes.variables[time_var].setncattr('calendar',data_grp.variables[time_var].calendar)

        if not 's'+gotvar.capitalize() in output_fluxes.dimensions.keys():
            length=dist_def['lengths'][dist_def['vars'].index(gotvar)]
            output_fluxes.createDimension('s'+gotvar.capitalize(),length-1)
            output_fluxes.createVariable('s'+gotvar.capitalize(),'d',('s'+gotvar.capitalize(),))
            output_fluxes.variables['s'+gotvar.capitalize()][:]=disc.inv_conversion(gotvar)(disc.inv_discretization(gotvar)(np.arange(1,length)))
        for other_gotvar in fluxes.dtype.names:
            if (not other_gotvar.capitalize() in output_fluxes.dimensions.keys()):
                length=dist_def['lengths'][dist_def['vars'].index(other_gotvar)]
                output_fluxes.createDimension(other_gotvar.capitalize(),length)
                output_fluxes.createVariable(other_gotvar.capitalize(),'d',(other_gotvar.capitalize(),))

                output_fluxes.variables[other_gotvar.capitalize()][:]=disc.inv_conversion(other_gotvar)(disc.inv_discretization(other_gotvar)(np.arange(0,length)+0.5))
        
        phase_space_dims=tuple(['s'+ps_dim.capitalize() if ps_dim==gotvar else ps_dim.capitalize() for ps_dim in dist_def['vars']])
        out_flux_var=flux_var+'_'+var
        output_fluxes.createVariable(out_flux_var,'f',(time_var,)+phase_space_dims,zlib=options.compression)

        shape=list(output_fluxes.variables[out_flux_var].shape)
        index_staggered = list(output_fluxes.variables[out_flux_var].dimensions).index('s'+gotvar.capitalize())
        shape[index_staggered]+=1
        temp=np.take(
                    np.reshape(fluxes[gotvar],tuple(shape),order='F'),
                        range(1,shape[index_staggered]),axis=index_staggered)
        output_fluxes.variables[out_flux_var][:]=temp
        output.sync()

        if options.divergence and not 'weight' in out_flux_var.split('_'):
            phase_space_dims=tuple([ps_dim.capitalize() for ps_dim in dist_def['vars']])
            output_fluxes.createVariable(out_flux_var+'_diff','f',(time_var,)+phase_space_dims,zlib=options.compression)

            shape_cat = copy.copy(shape)
            shape_cat[index_staggered]=1

            shape=list(output_fluxes.variables[out_flux_var+'_diff'].shape)
            index_staggered = list(output_fluxes.variables[out_flux_var+'_diff'].dimensions).index(gotvar.capitalize())
            shape_cat = copy.copy(shape)
            shape_cat[index_staggered]=1
            output_fluxes.variables[out_flux_var+'_diff'][:]=np.diff(
                            np.concatenate(
                            (
                            np.reshape(fluxes[gotvar],tuple(shape),order='F'),
                            np.zeros(shape_cat),
                            ),
                            axis=index_staggered
                            ),axis=index_staggered)
            output.sync()
    return

def output_conversion_masstendency(data,output,source_group_name,flux_var,var,mass,options,dist_def,source_name,out_name):
    out_group_name='masstendency_'+out_name+var

    if not out_group_name in output.groups.keys():
        output.createGroup(out_group_name)
    output_mass=output.groups[out_group_name]

    if source_group_name in data.groups.keys():
        data_grp=data.groups[source_group_name]
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

    disc=discretizations_class(dist_def)
    for other_gotvar in dist_def['vars']:
        if (not other_gotvar.capitalize() in output_mass.dimensions.keys()):
            length=dist_def['lengths'][dist_def['vars'].index(other_gotvar)]
            output_mass.createDimension(other_gotvar.capitalize(),length)
            output_mass.createVariable(other_gotvar.capitalize(),'d',(other_gotvar.capitalize(),))

            output_mass.variables[other_gotvar.capitalize()][:]=disc.inv_conversion(other_gotvar)(disc.inv_discretization(other_gotvar)(np.arange(0,length)+0.5))
    
    phase_space_dims=tuple([ps_dim.capitalize() for ps_dim in dist_def['vars']])
    output_mass.createVariable(flux_var,'f',(time_var,)+phase_space_dims,zlib=options.compression)

    shape=list(mass.shape[:-1]+output_mass.variables[flux_var].shape[-len(dist_def['vars']):])

    output_mass.variables[flux_var][:]=np.sum(
                                    np.reshape(mass,tuple(shape),order='F'),axis=tuple(range(1,len(mass.shape)-1)))
    output.sync()

    if options.divergence and not 'weight' in flux_var.split('_'):
        phase_space_dims=tuple([ps_dim.capitalize() for ps_dim in dist_def['vars']])
        output_mass.createVariable(flux_var+'_diff','f',(time_var,)+phase_space_dims,zlib=options.compression)
        shape=list(mass.shape[:-1]+output_mass.variables[flux_var].shape[-len(dist_def['vars']):])
        shape[0]-=1

        output_mass.variables[flux_var+'_diff'][:-1,...]=np.sum(
                                            np.reshape(
                                            mass[1:,...]-mass[:-1,...]
                                            ,tuple(shape),order='F')
                                            ,axis=tuple(range(1,len(mass.shape)-1)))
        output_mass.variables[flux_var+'_diff'][-1,...]=0.0
        output.sync()
    return

def output_conversion_sources(data,output,source_group_name,flux_var,var,sources,options,dist_def,source_name,out_name):
    out_group_name='sources_'+out_name+var

    if not out_group_name in output.groups.keys():
        output.createGroup(out_group_name)
    output_sources=output.groups[out_group_name]

    if source_group_name in data.groups.keys():
        data_grp=data.groups[source_group_name]
    else:
        data_grp=data.groups[var]
    for time_type in ['time','stime']:
        if time_type in data_grp.dimensions.keys():
            time_var=time_type

    if not time_var in output_sources.dimensions.keys():
        output_sources.createDimension(time_var,len(data_grp.variables[time_var]))
        output_sources.createVariable(time_var,'d',(time_var,))
        output_sources.variables[time_var][:]=data_grp.variables[time_var][:]
        output_sources.variables[time_var].setncattr('units',data_grp.variables[time_var].units)
        output_sources.variables[time_var].setncattr('calendar',data_grp.variables[time_var].calendar)

    disc=discretizations_class(dist_def)
    for other_gotvar in dist_def['vars']:
        if (not other_gotvar.capitalize() in output_sources.dimensions.keys()):
            length=dist_def['lengths'][dist_def['vars'].index(other_gotvar)]
            output_sources.createDimension(other_gotvar.capitalize(),length)
            output_sources.createVariable(other_gotvar.capitalize(),'d',(other_gotvar.capitalize(),))

            output_sources.variables[other_gotvar.capitalize()][:]=disc.inv_conversion(other_gotvar)(disc.inv_discretization(other_gotvar)(np.arange(0,length)+0.5))
    
    phase_space_dims=tuple([ps_dim.capitalize() for ps_dim in dist_def['vars']])
    output_sources.createVariable(flux_var,'f',(time_var,)+phase_space_dims,zlib=options.compression)

    shape=list(sources.shape[:-1]+output_sources.variables[flux_var].shape[-len(dist_def['vars']):])

    output_sources.variables[flux_var][:]=np.sum(
                                    np.reshape(sources,tuple(shape),order='F'),axis=tuple(range(1,len(sources.shape)-1)))
    output.sync()
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

def create_masstendency_science(data,source_group_name,flux_var,var,dist_def,quantity=''):
    disc=discretizations_class(dist_def[var])

    list_types=create_structured_array(data.groups[source_group_name],flux_var,dist_def[var])
    fluxes =collapse_dims_last(data.groups[source_group_name],flux_var,dist_def[var]['dims'])

    jd_space=np.zeros(fluxes.shape,dtype=list_types)
    jd_space['flux']=fluxes

    dimensions=[dim[1:] if dim[0]=='s' else dim for dim in dist_def[var]['dims'] ]

    if dist_def[var]['time_type']=='previous':
        for gotvar in dist_def[var]['vars']:
            jd_space[gotvar]=disc.discretization(gotvar)(disc.conversion(gotvar)(
                                collapse_dims_last_var(data.groups[gotvar].variables[gotvar][:-1,...],
                                                     data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
                                        ))
    elif dist_def[var]['time_type']=='full':
        for gotvar in dist_def[var]['vars']:
            jd_space[gotvar]=disc.discretization(gotvar)(disc.conversion(gotvar)(
                                collapse_dims_last_var(data.groups[gotvar].variables[gotvar][:],
                                                     data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
                                        ))
    if quantity!='':
        jd_space['flux']*=(
                            collapse_dims_last_var(data.groups[quantity].variables[quantity][:-1,...],
                                                 data.groups[quantity].variables[quantity].dimensions ,dimensions)
                         )
    return jd_space

def create_sources_science(data,source_group_name,flux_var,var,dist_def,quantity=''):
    disc=discretizations_class(dist_def[var])

    list_types=create_structured_array(data.groups[source_group_name],flux_var,dist_def[var])
    fluxes =collapse_dims_last(data.groups[source_group_name],flux_var,dist_def[var]['dims'])

    jd_space=np.zeros(fluxes.shape,dtype=list_types)
    jd_space['flux']=fluxes

    dimensions=[dim[1:] if dim[0]=='s' else dim for dim in dist_def[var]['dims'] ]

    if dist_def[var]['time_type']=='centre':
        for gotvar in dist_def[var]['vars']:
            jd_space[gotvar]=disc.discretization(gotvar)(disc.conversion(gotvar)(
                                collapse_dims_last_var(data.groups[gotvar].variables[gotvar][1:,...],
                                                     data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
                                        ))
    elif dist_def[var]['time_type']=='full':
        for gotvar in dist_def[var]['vars']:
            jd_space[gotvar]=disc.discretization(gotvar)(disc.conversion(gotvar)(
                                collapse_dims_last_var(data.groups[gotvar].variables[gotvar][:],
                                                     data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
                                        ))
    if quantity!='':
        jd_space['flux']*=(
                            collapse_dims_last_var(data.groups[quantity].variables[quantity][:-1,...],
                                                 data.groups[quantity].variables[quantity].dimensions ,dimensions)
                         )
    return jd_space

def create_massfluxes_science(data,group_name,flux_var,var,dist_def,options,quantity=''):
    disc=discretizations_class(dist_def[var])

    list_types=create_structured_array(data.groups[group_name],flux_var,dist_def[var])
    fluxes =collapse_dims_last(data.groups[group_name],flux_var,dist_def[var]['dims'])

    jd_space=np.zeros(fluxes.shape,dtype=list_types)
    jd_space['flux']=fluxes

    dimensions=[dim[1:] if dim[0]=='s' else dim for dim in dist_def[var]['dims'] ]

    checks=dict()
    for gotvar in dist_def[var]['vars']:
        gotvar_n, gotvar_p=values_diff_c_grid(data,dist_def[var],gotvar,dimensions)
        if ('check' in dir(options) and options.check!=None): 
            for checkvar in options.check:
                if not gotvar in checks.keys():
                    checks[gotvar]=[]
                checkvar_n, checkvar_p=values_diff_c_grid(data,dist_def[var],checkvar,dimensions)
                checks[gotvar].append((fluxes*(gotvar_n-gotvar_p)*(0.5*(checkvar_n+checkvar_p))).sum(-1))
        if quantity!='':
            quantityvar_n, quantityvar_p=values_diff_c_grid(data,dist_def[var],quantity,dimensions)
            jd_space['flux']*=0.5*(quantityvar_n+quantityvar_p)

        gotvar_p=disc.discretization(gotvar)(disc.conversion(gotvar)(gotvar_p))
        gotvar_n=disc.discretization(gotvar)(disc.conversion(gotvar)(gotvar_n))

        if not 'weight' in flux_var.split('_'):
            jd_space[gotvar]=gotvar_p
            jd_space[gotvar+'_MASK']=np.sign(gotvar_n-gotvar_p)
            jd_space[gotvar+'_TOTAL']=gotvar_n-gotvar_p
            jd_space[gotvar+'_SUM']=np.zeros(jd_space[gotvar+'_MASK'].shape)
        else:
            jd_space[gotvar]=np.minimum(gotvar_p,gotvar_n)
            jd_space[gotvar+'_MASK']=np.abs(np.sign(gotvar_n-gotvar_p))
            jd_space[gotvar+'_TOTAL']=np.abs(gotvar_n-gotvar_p)
            jd_space[gotvar+'_SUM']=np.zeros(jd_space[gotvar+'_MASK'].shape)

    return jd_space, checks

def values_diff_c_grid(data,dist_def,gotvar,dimensions):
    if dist_def['bc']=='staggered':
        gotvar_n=collapse_dims_last_var(
                                        np.take(data.groups[gotvar].variables[gotvar],
                                                np.arange(data.groups[gotvar].variables[gotvar].shape[dist_def['axis']])[1:],
                                                axis=dist_def['axis']),
                                         data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
        gotvar_p=collapse_dims_last_var(
                                        np.take(data.groups[gotvar].variables[gotvar],
                                                np.arange(data.groups[gotvar].variables[gotvar].shape[dist_def['axis']])[:-1],
                                                axis=dist_def['axis']),
                                         data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
    elif dist_def['bc']=='constant':
        gotvar_n=collapse_dims_last_var(
                                        data.groups[gotvar].variables[gotvar][:],
                                         data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
        gotvar_p=gotvar_n
    elif dist_def['bc']=='periodic':
        gotvar_n=collapse_dims_last_var(data.groups[gotvar].variables[gotvar][:],
                                            data.groups[gotvar].variables[gotvar].dimensions ,dimensions)
        gotvar_p=collapse_dims_last_var(np.roll(data.groups[gotvar].variables[gotvar][:],1,axis=dist_def['axis']),
                                            data.groups[gotvar].variables[gotvar].dimensions ,dimensions)

    if dist_def['axis']!=0 and dist_def['time_type']!='full':
        gotvar_n=gotvar_n[1:,...]
        gotvar_p=gotvar_p[1:,...]
    return gotvar_n, gotvar_p

def got_digitize_floor(array,length):
    out=np.rint(np.floor(array))
    out[out<0]=0
    out[out>length-1]=length-1
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
                #for gotvar in dist_def['vars']:
                #    jd_out[gotvar]+=jd_out_tmp[gotvar]
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

    #CREATE OUPTUT:
    #list_types=[(gotvar,np.float) for gotvar in dist_def['vars']]
    #jd_out=np.zeros((dist_def['phase_space_length']+1,),list_types)

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
            if att=='history':
                att_val+='\n' 
                att_val+='joint_distribution '+pkg_resources.get_distribution('pydiv').version
            if 'encode' in dir(att_val):
                att_val=att_val.encode('ascii','replace')
	    setattr(output,att,att_val)
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
              'smoist': ('time','lev','lat','lon'),
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
        temp=output.groups[var].createVariable(var,'f',var_list[var])
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

    var='smoist'
    output.groups[var].variables[var][:]=((220+np.random.rand(*output.groups[var].variables[var].shape)*9)/273.15-1.0)*1001

    var='hus'
    output.groups[var].variables[var][:]=np.random.rand(*output.groups[var].variables[var].shape)*4*1001/2.5e6

    var='pa'
    output.groups[var].variables[var][:]=100e2+np.random.rand(*output.groups[var].variables[var].shape)*900e2

    for var in ['smoist','hus','pa']:
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
    science_parser.add_argument('-c','--check',action='append', type=str, choices=['ta','gh_gc','gc_gv','gd','thetav'],
                                       help='Compute de global derivatives of the massfluxes multiplied by the check variables.' )

    test_parser=subparsers.add_parser('transform_test',
                                           description=textwrap.dedent(
                                                '''Compute the thermodynamics transform of Laliberte et al. (2015).
                                                 '''),epilog=epilog
                                         )
    test_parser.add_argument("out_netcdf_file",help="netCDF output file")

    thermal_parser=subparsers.add_parser('transform_thermal',
                                           description=textwrap.dedent(
                                                '''Project the thermodynamics transform on sub-coordinates.
                                                 '''),epilog=epilog
                                         )
    generate_subparser(thermal_parser)
    thermal_parser.add_argument("--divergence",dest="divergence",
                      default=False, action="store_true",
                      help="Compute divergence in phase space")
    thermal_parser.add_argument("--coordinates",dest="coordinates",
                      choices=['smoist,ta','hus,gh_gc','hus,gc_gv','hus,gd','pa,thetav'],
                      default='smoist,ta',
                      help="coordinates onto which the projection should be performed")
    thermal_parser.add_argument('-c','--check',action='append', type=str, choices=['ta','gh_gc','gc_gv','gd','thetav'],
                                       help='Compute the global derivatives of the massfluxes multiplied by the check variables.' )

    coarse_parser=subparsers.add_parser('coarse_grain',
                                           description=textwrap.dedent(
                                                '''Coarse grain fluxes.
                                                 '''),epilog=epilog
                                         )
    generate_subparser(coarse_parser)
    coarse_parser.add_argument("--divergence",dest="divergence",
                      default=False, action="store_true",
                      help="Compute divergence in phase space")
    coarse_parser.add_argument('-q','--quantity',action='append', type=str, choices=['ke','pe','hmoist','hus','ta','pa','expansion'],
                                       help='Coarse grain to C-grid.' )

    options=parser.parse_args()

    if options.command == "transform_science":
        heat_engine_science(options)
    elif options.command == "transform_test":
        generate_test(options)
    elif options.command == "transform_thermal":
        projection_science(options)
    elif options.command == "coarse_grain":
        coarse_grain(options)
    return
      
if __name__ == "__main__":
    main()
