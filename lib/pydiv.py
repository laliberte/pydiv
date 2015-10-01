from . import spherical_tools
from netCDF4 import Dataset
import numpy as np

def replicate_netcdf_file(output,data):
    #This function replicates a netcdf file
    for att in data.ncattrs():
        att_val=getattr(data,att)
        if 'encode' in dir(att_val):
            att_val=att_val.encode('ascii','replace')
        setattr(output,att,att_val)
    output.sync()
    return output

def replicate_netcdf_var(output,data,var):
    #This function replicates a netcdf variable 
    for dims in data.variables[var].dimensions:
        if dims not in output.dimensions.keys():
            output.createDimension(dims,len(data.dimensions[dims]))
            dim_var = output.createVariable(dims,'d',(dims,))
            dim_var[:] = data.variables[dims][:]
            output = replicate_netcdf_var(output,data,dims)

    if var not in output.variables.keys():
        output.createVariable(var,'d',data.variables[var].dimensions,zlib=True)
    for att in data.variables[var].ncattrs():
        att_val=getattr(data.variables[var],att)
        if att[0]!='_':
            if 'encode' in dir(att_val):
                att_val=att_val.encode('ascii','replace')
            setattr(output.variables[var],att,att_val)
    output.sync()
    return output

def multiply_by_area(options):
    data=Dataset(options.in_file)
    output=Dataset(options.out_file,'w')
    replicate_netcdf_file(output,data)
    lengths=spherical_tools.coords(data)

    for var in ['wa','mass','weight_wa']:
        replicate_netcdf_var(output,data,var)
        output.variables[var][:]=data.variables[var][:]*np.reshape(lengths.area_lat_lon,(1,1,)+data.variables[var].shape[-2:])
        output.sync()
    data.close()
    return output

def multiply_by_length(options,output):
    data=Dataset(options.in_file)
    lengths=spherical_tools.coords(data)

    var='ua'
    for var_sub in [var,'weight_'+var]:
        replicate_netcdf_var(output,data,var_sub)
        output.variables[var_sub][:]=data.variables[var_sub][:]*np.reshape(lengths.mer_len_lat_slon,(1,1,)+(data.variables[var_sub].shape[-2],)+(1,))
    output.sync()

    var='va'
    for var_sub in [var,'weight_'+var]:
        replicate_netcdf_var(output,data,var_sub)
        output.variables[var_sub][:]=data.variables[var_sub][:]*np.reshape(lengths.zon_len_slat_lon,(1,1,)+data.variables[var_sub].shape[-2:])
    output.sync()
    return output

def wa_from_div(options):
    data=Dataset(options.in_file)
    output=Dataset(options.out_file,'w')
    replicate_netcdf_file(output,data)

    #Retrieve data and create output:
    vars_space=dict()
    for var in ['div','wa']:
        if var=='wa': replicate_netcdf_var(output,data,var)
        vars_space[var]=data.variables[var][:,:,:,:].astype(np.float,copy=False)
    for var in ['mass']:
        vars_space[var]=(data.variables[var][1:,:,:,:].astype(np.float,copy=False) -
                         data.variables[var][:-1,:,:,:].astype(np.float,copy=False))
    
    data.close()
    
    #Compute the mass divergence:
    DIV = vars_space['mass'] + vars_space['div']
    vars_space['wa'][:,1:,...]=-np.cumsum(np.ma.array(DIV).anom(1),axis=1) 
    #vars_space['wa'][1:-1,1:,:]=np.ma.array(DIV).anom(0) 
    vars_space['wa'][:,0,...]=0.0
    for var in ['wa']:
        output.variables[var][:]=vars_space[var]

    output.sync()
    output.close()
    return


def correct_mass_fluxes(options):
    data=Dataset(options.in_file)
    output=Dataset(options.out_file,'w')
    replicate_netcdf_file(output,data)

    #Retrieve data and create output:
    type=np.float
    vars_space=dict()
    for var in ['ua','va','wa']:
        replicate_netcdf_var(output,data,var)
        vars_space[var]=data.variables[var][:].astype(type,copy=False)
    for var in ['mass']:
        replicate_netcdf_var(output,data,var)
        vars_space[var]=data.variables[var][:].astype(type,copy=False)
        output.variables[var][:]=vars_space[var]
    for var in ['dmassdt']:
        vars_space[var]=(vars_space['mass'][1:,...]-vars_space['mass'][:-1,...]).astype(type,copy=False)
    
    
    #Compute spherical lengths:
    lengths=spherical_tools.coords(data)
    #Create vector calculus space:
    vector_calculus=spherical_tools.vector_calculus_spherical(vars_space['dmassdt'].shape[1:],lengths)

    for id in [0,-1]:
        if np.abs(data.variables['lat'][id])==90.0:
            vars_space['ua'][:,:,id,:]=0.0
    
    #Compute the mass divergence:
    DIV=np.zeros_like(vars_space['dmassdt'])
    for time_id, time in enumerate(range(len(data.variables['time']))):
        DIV[time_id,...] = (vars_space['dmassdt'][time_id,...] + 
                vector_calculus.DIV_from_UVW_mass(*[vars_space[var][time_id,...] for var in ['ua','va','wa']])
                )

    for time_id, time in enumerate(range(len(data.variables['time']))):
        #Compute the velocity potential of the residual:
        Chi = vector_calculus.inverse_laplacian(-DIV[time_id,...],maxiter=options.maxiter)

        #Compute the velocities corrections and record to output:
        for var, correction in zip(['ua','va','wa'],vector_calculus.UVW_mass_from_Chi(Chi)):
            vars_space[var][time_id,...]-=correction

    dmass=np.zeros_like(vars_space['dmassdt'])
    for time_id in range(len(data.variables['time'])):
        dmass[time_id,...] = (vars_space['dmassdt'][time_id,...] + 
                        vector_calculus.DIV_from_UVW_mass(*[vars_space[var][time_id,...] for var in ['ua','va','wa']])
                             )

    #Fix vertical velocity:
    vars_space['wa'][:,1:-1,...]-=np.cumsum(np.ma.array(dmass).anom(1),axis=1)[:,:-1,:,:]

    for var in ['ua','va','wa']:
        output.variables[var][:]=vars_space[var]

    if options.check_output:
        output.createVariable('dmass_old',type,('time','lev','lat','lon'))
        output.variables['dmass_old'][:,...] = DIV

        output.createVariable('dmass',type,('time','lev','lat','lon'))
        for time_id in range(len(data.variables['time'])):
            dmass[time_id,...] = (vars_space['dmassdt'][time_id,...] + 
                                                        vector_calculus.DIV_from_UVW_mass(*[vars_space[var][time_id,...] for var in ['ua','va','wa']])
                                                        )
        output.variables['dmass'][:,...] = dmass

    output.sync()
    output.close()
    data.close()

    return

def coarse_grain_horizontal(options):
    data=Dataset(options.in_file)
    output=Dataset(options.out_file,'w')
    replicate_netcdf_file(output,data)

    lengths_high=spherical_tools.coords(data)

    for var in ['time','lev','slev']:
        replicate_netcdf_var(output,data,var)

    output.createDimension('slon',len(data.dimensions['slon'])/2)
    output.createVariable('slon',np.float,('slon',))
    output.variables['slon'][:]=data.variables['slon'][::2]
    output.createDimension('lon',len(data.dimensions['lon'])/2)
    output.createVariable('lon',np.float,('lon',))
    output.variables['lon'][:]=(data.variables['lon'][::2]+data.variables['lon'][1::2])/2

    output.createDimension('slat',len(data.dimensions['slat'])/3)
    output.createVariable('slat',np.float,('slat',))
    output.variables['slat'][:]=data.variables['slat'][1::3]
    output.createDimension('lat',len(data.dimensions['slat'])/3+1)
    output.createVariable('lat',np.float,('lat',))
    output.variables['lat'][1:-1]=(data.variables['lat'][2:-2:3]+data.variables['lat'][3:-2:3]+data.variables['lat'][4:-2:3])/3
    output.variables['lat'][0]=-90.0
    output.variables['lat'][-1]=90.0

    lengths_low=spherical_tools.coords(output)
    
    var='ua'
    output.createVariable(var,np.float,('time','lev','lat','slon'))
    output.variables[var][:,:,1:-1,:]=(data.variables[var][:,:,2:-2:3,::2]+data.variables[var][:,:,3:-2:3,::2]+data.variables[var][:,:,4:-2:3,::2])
    output.variables[var][:,:,0,:]=0.0
    output.variables[var][:,:,-1,:]=0.0

    var='va'
    output.createVariable(var,np.float,('time','lev','slat','lon'))
    output.variables[var][:]=(data.variables[var][:,:,1::3,::2]+data.variables[var][:,:,1::3,1::2])
    output.sync()

    for var in ['wa']:
        output.createVariable(var,np.float,('time','slev','lat','lon'))
        output.variables[var][:]=full_average(data.variables[var][:],output.variables[var].shape)
    output.sync()

    for var in ['mass']:
        output.createVariable(var,np.float,('time','lev','lat','lon'))
        output.variables[var][:]=full_average(data.variables[var][:],output.variables[var].shape)
    output.sync()

    for var in ['ta','hus','pa']:
        output.createVariable(var,np.float,('time','lev','lat','lon'))
        #temp=data.variables[var][:]*np.reshape(lengths_high.area_lat_lon,(1,1,)+lengths_high.area_lat_lon.shape)*data.variables['dpa'][:]
        #output.variables[var][:]=full_average(temp,output.variables[var].shape)/(output.variables['dpa']*np.reshape(lengths_low.area_lat_lon,(1,1,)+lengths_low.area_lat_lon.shape))
        output.variables[var][:]=full_average(data.variables[var][:]*data.variables['mass'][:],output.variables[var].shape)/output.variables['mass']
    output.sync()

    test_divergence=False
    if test_divergence:
        #Retrieve data and create output:
        vars_space=dict()
        for var in ['ua','va','wa']:
            vars_space[var]=output.variables[var][0,:,:,:].astype(np.float,copy=False)

        vars_space['dmassdt']=(output.variables['mass'][1,:,:,:].astype(np.float,copy=False)-
                               output.variables['mass'][0,:,:,:].astype(np.float,copy=False)) 

        #Compute spherical lengths:
        lengths=spherical_tools.coords(output)
        #Create vector calculus space:
        vector_calculus=spherical_tools.vector_calculus_spherical(vars_space['dpadt'].shape,lengths)

        #FOR MERRA:
        vars_space['ua']/=lengths.mer_len_lat_slon
        vars_space['va']/=lengths.zon_len_slat_lon
        vars_space['wa']/=np.reshape(lengths.area_lat_lon,(1,)+ vars_space['wa'].shape[1:])
        ####

        #Compute the mass divergence:
        DIV = vars_space['dmassdt'] + vector_calculus.DIV_from_UVW(*[vars_space[var] for var in ['ua','va','wa']])

        output.createVariable('dmass',np.float,('time','lev','lat','lon'))
        output.variables['dmass'][:]=DIV
    output.close()
    data.close()
    return

def full_average(array,shape):
    temp_out=np.empty(shape)
    temp_out[:,:,1:-1,:]=(
          (array[:,:,2:-2:3,::2]+array[:,:,3:-2:3,::2]+array[:,:,4:-2:3,::2]) +
          (array[:,:,2:-2:3,1::2]+array[:,:,3:-2:3,1::2]+array[:,:,4:-2:3,1::2])
          )
    temp_out[:,:,0,:] = np.reshape(array[:,:,:2,:].sum(-1).sum(-1)/temp_out.shape[-1],array.shape[:2]+(1,))
    temp_out[:,:,-1,:] = np.reshape(array[:,:,-2:,:].sum(-1).sum(-1)/temp_out.shape[-1],array.shape[:2]+(1,))
    return temp_out


def main():
    import argparse 
    import textwrap

    #Option parser
    description=textwrap.dedent('''\
    This script fixes the mass continuity equation.
    Input file must be in C-grid format on a lat-lon grid
    with the pole included (odd number of grid points in the latitude)
    and contain:
    ua, va : the mass fluxes across the grid boundary
    wa : the pressure velocity
    mass : the mass of the atmospheric layer
    ''')
    epilog='Frederic Laliberte, Paul Kushner 10/2013'
    version_num='0.1'
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                            description=description,
                            version='%(prog)s '+version_num,
                            epilog=epilog)
    subparsers = parser.add_subparsers(help='commands',dest='command')

    correct_parser=subparsers.add_parser('correct_mass_fluxes',
                                           help='This function makes mass fluxes mass conserving.',
                                           epilog=epilog,
                                           formatter_class=argparse.RawTextHelpFormatter)
    correct_parser.add_argument('--maxiter',type=int,default=10,help='Number of iterations')
    correct_parser.add_argument('--check_output',default=False,action='store_true',help='Outputs the mass conservation.')
    input_arguments(correct_parser)

    wa_parser=subparsers.add_parser('wa_from_div',
                                           help='This function computes wa from the divergence of velocities.',
                                           epilog=epilog,
                                           formatter_class=argparse.RawTextHelpFormatter)
    input_arguments(wa_parser)
    
    coarse_parser=subparsers.add_parser('coarse_grain',
                                           help='This function coarse grains.',
                                           epilog=epilog,
                                           formatter_class=argparse.RawTextHelpFormatter)
    input_arguments(coarse_parser)
    
    weight_parser=subparsers.add_parser('weight_by_area',
                                           help='This function weights dpa and wa by area.',
                                           epilog=epilog,
                                           formatter_class=argparse.RawTextHelpFormatter)
    input_arguments(weight_parser)
    weight2_parser=subparsers.add_parser('weight_by_area_and_length',
                                           help='This function weights dpa and wa by area and ua, va by lengths.',
                                           epilog=epilog,
                                           formatter_class=argparse.RawTextHelpFormatter)
    input_arguments(weight2_parser)

    options=parser.parse_args()

    if options.command=='correct_mass_fluxes':
        correct_mass_fluxes(options)
    elif options.command=='wa_from_div':
        wa_from_div(options)
    elif options.command=='coarse_grain':
        coarse_grain_horizontal(options)
    elif options.command=='weight_by_area':
        output=multiply_by_area(options)
        output.close()
    elif options.command=='weight_by_area_and_length':
        output=multiply_by_area(options)
        output=multiply_by_length(options,output)
        output.close()
        
    return 

def input_arguments(parser):
    parser.add_argument('in_file',
                                 help='Input file')
    parser.add_argument('out_file',
                                 help='Output file')
    return

if __name__ == "__main__":
    main()
