"""
Frederic Laliberte 12/2013
"""
import sys
import numpy as np
from netCDF4 import Dataset
import datetime as dt
import copy
import pkg_resources

global bins_name

bins_name='bins'

def create_jd_output(data,nc_output_file):
    #CREATE THE OUTPUT FILE
    output = Dataset(nc_output_file+'.tmp','w',format='NETCDF4')
    output = replicate_netcdf_file(output,data)

    dist_def=dict()

    #CHECK THE SPECIFICATION OF JOINT DISTRIBUTION
    if 'joint_dist_dims' in data.ncattrs():
       dist_def['dims'] = getattr(data,'joint_dist_dims').split(',')
    else:
       raise IOError('input file must have a joint_dist_dims global attribute')

    if 'joint_dist_vars' in data.ncattrs():
       dist_def['vars'] = [ var.encode('ascii','ignore') for var in getattr(data,'joint_dist_vars').split(',') ]
    else:
       raise IOError('input file must have a joint_dist_vars global attribute')
    

    #CONVERT THE DIST VARS TO BIN VALUES AND PUT THEM ALONG THE SAME AXIS
    dist_def['lengths'] = [getattr(data,'jd_dist_'+var) for var in dist_def['vars']]

    #COMPUTE THE SIZE OF THE PHASE SPACE:
    dist_def['phase_space_length']=compute_phase_space_length(dist_def['lengths'])

    if bins_name not in output.dimensions.keys():
        #CREATE SINGLE DISTRIBUTION AXIS:
        output.createDimension(bins_name,dist_def['phase_space_length']+1)
        temp_dim = output.createVariable(bins_name,'f',(bins_name,))
        temp_dim[:] = np.array(range(dist_def['phase_space_length']+1))
        #DESCRIBE JD:
        output.variables[bins_name].jd_dims=','.join([str(var) for var in dist_def['vars']])
        output.variables[bins_name].jd_dims_length=dist_def['lengths']
        #PUT JD DESCRIPTION FROM DIST VARS INTO BINS DESCRIPTION:
        #for var in dist_def['vars']:
        #    setattr(output.variables[bins_name],var,getattr(data,'jd_dist_'+var))

        #ADD STANDARD ATTRIBUTES TO BINS DIMENSION
        output.variables[bins_name].long_name='joint distribution bins'
    output.sync()
    return output, dist_def

def joint_distribution_got(nc_input_file,nc_output_file,nc_comp):
    #LOAD THE DATA:
    data = Dataset(nc_input_file,'r')

    #CREATE OUTPUT FILE
    output, dist_def = create_jd_output(data,nc_output_file)

    var_list = data.variables.keys()
    #DETERMINE WHICH VARIABLES TO DISTRIBUTE
    for var in var_list:
        if 'joint_dist' in data.variables[var].ncattrs():
	   print(var)
           output=compute_jd_got(data,output,dist_def,var,nc_comp)
           output.sync()
    data.close()
    output.close()
    import os
    os.rename(nc_output_file+'.tmp',nc_output_file)
    return

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

def compute_jd_got(data,output,dist_def,var,nc_comp):
       #VERIFY THAT ALL THE JD INFO IS AVAILABLE:
       output,var_dims=add_dimension(data,output,dist_def,var)

       #CREATE A STRUCTURED ARRAY:
       list_types=[('flux',np.float64)]
       for gotvar in dist_def['vars']:
            list_types.append((gotvar,np.float64))
            list_types.append((gotvar+'_MASK',np.int8))
            list_types.append((gotvar+'_TOTAL',np.int32))
            list_types.append((gotvar+'_SUM',np.float))

       #LOAD THE DATA:
       fluxes =collapse_dims_last(data,var,dist_def['dims'])
       jd_space=np.zeros(fluxes.shape,dtype=list_types)
       jd_space['flux']=fluxes
       for gotvar in dist_def['vars']:
            jd_space[gotvar]=collapse_dims_last(data,gotvar,dist_def['dims'])
            jd_space[gotvar+'_MASK']=collapse_dims_last(data,gotvar+'_DIFF_MASK',dist_def['dims'])
            jd_space[gotvar+'_TOTAL']=collapse_dims_last(data,gotvar+'_DIFF',dist_def['dims'])
            jd_space[gotvar+'_SUM']=np.zeros(jd_space[gotvar+'_MASK'].shape)
            dist_def[gotvar+'_TEST']=0
       #print [ (jd_space[key],key) for key in list(jd_space.dtype.names)]

       #dist_def['EPT_TEST']=3
       #dist_def['PRES_TEST']=1
       #print compute_tests('QPT',dist_def,jd_space,output)

       temp_out = np.apply_along_axis(compute_vector_jd_got,-1,
                                         jd_space,
                                         dist_def,
                                         var)

       #CREATE OUTPUT VARIABLE
       for gotvar in temp_out.dtype.names:
           temp_ptr=temp_out[gotvar]
           final_jd = output.createVariable(var+'_'+gotvar,'d',tuple(var_dims),zlib=nc_comp)
           output = replicate_netcdf_var2(output,data,var,var+'_'+gotvar)
           final_jd[:]=temp_ptr
       return output  

def got_digitize(array,length):
    array-=0.25
    array=np.rint(array)
    array[array<0]=0
    array[array>length-1]=length-1
    return array

def compute_vector_jd_got(jd_space,dist_def,var):
    #CREATE OUPTUT:
    list_types=[(gotvar,np.float32) for gotvar in dist_def['vars']]
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
    bin_array = np.zeros_like(jd_space['flux'][binning_mask], dtype=int)
    for nv,dvar in enumerate(dist_def['vars']):
        bin_array+=np.prod(dist_def['lengths'][0:nv])*\
                           got_digitize(jd_space[dvar][binning_mask]+jd_space[dvar+'_SUM'][binning_mask],dist_def['lengths'][nv])
    if dist_def['phase_space_length'] < bin_array.max(): raise IOError('number of actual bins is larger than expected output')

    #SETUP THE INVERSION
    temp_data = np.abs(jd_space['flux'][binning_mask])*jd_space[gotvar+'_MASK'][binning_mask]

    #DO THE COMPUTATION USING NP.BINCOUNT
    jd_out[gotvar] += np.bincount(bin_array,
                                 weights=temp_data,
                                 minlength=dist_def['phase_space_length']+1)

    #APPLY THE GRADIENT:
    jd_space[gotvar+'_SUM'][binning_mask]+=0.5*jd_space[gotvar+'_MASK'][binning_mask]
    jd_space[gotvar+'_TOTAL'][binning_mask]-=jd_space[gotvar+'_MASK'][binning_mask]
    jd_space[gotvar+'_MASK'][binning_mask]*=np.where(np.abs(jd_space[gotvar+'_TOTAL'][binning_mask])>0,1,0)
    return jd_space,jd_out

#VARIOUS TESTS

def compute_binomial(x,m,k,position):
    """
    Computes \sum_{n=0}^{m-1}(x+n)^k for float x
    """
    shape=tuple(1 for val in x.shape)
    if position=='after':
        M=np.sign(np.reshape(m,m.shape+(1,)))*np.reshape(np.array(range(0,np.abs(m).max())),shape+(np.abs(m).max(),))
        M=np.where(np.abs(M)>=np.reshape(np.abs(m),m.shape+(1,)),
                   -np.reshape(x,x.shape+(1,)),M)
    elif position=='before':
        M=np.sign(np.reshape(m,m.shape+(1,)))*np.reshape(np.array(range(1,np.abs(m).max()+1)),shape+(np.abs(m).max(),))
        M=np.where(np.abs(M)>np.reshape(np.abs(m),m.shape+(1,)),
                   -np.reshape(x,x.shape+(1,)),M)
    return  ( (np.reshape(x,x.shape+(1,))+M)**k ).sum(-1)

def compute_tests(var_diff,dist_def,jd_space,output):
    #This test works with the simplest upwind discretization
    test=np.ones_like(jd_space['flux'])*np.abs(jd_space['flux'])
    for var in [ gotvar for gotvar in dist_def['vars'] if dist_def[gotvar+'_TEST']>0]:
        if dist_def['vars'].index(var)>dist_def['vars'].index(var_diff):
            test*= obtain_mid_point(jd_space[var_diff+'_TOTAL'],
                                    jd_space[var+'_TOTAL'],
                                    jd_space[var],
                                    dist_def[var+'_TEST'],'after')
        elif dist_def['vars'].index(var)<dist_def['vars'].index(var_diff):
            test*= obtain_mid_point(jd_space[var_diff+'_TOTAL'],
                                    jd_space[var+'_TOTAL'],
                                    jd_space[var],
                                    dist_def[var+'_TEST'],'before')
            
    return (test*jd_space[var_diff+'_TOTAL'].astype(np.float)).sum(-1)

def obtain_mid_point(diff_total,int_total,int_upwind,k,position):
    compute_binomial_vec=(lambda x,y:compute_binomial(x,y,k,position))
    mid_point=copy.copy(int_upwind)
    mid_point=np.where(np.abs(int_total)>=np.abs(diff_total),
                        compute_binomial_vec(int_upwind,diff_total),
                        mid_point)
    mid_point=np.where(np.abs(int_total)<np.abs(diff_total),
        (compute_binomial_vec(int_upwind,int_total)+(np.abs(diff_total)-np.abs(int_total))*(int_upwind+int_total.astype(np.float))**k),
                        mid_point)

    mid_point=np.where(np.abs(diff_total)>0,mid_point/np.abs(diff_total),mid_point)

    mid_point[np.abs(int_total)==0]=int_upwind[np.abs(int_total)==0]
    mid_point[np.abs(diff_total)==0]=int_upwind[np.abs(int_total)==0]
    mid_point[np.abs(diff_total)>0]/=np.abs(diff_total[np.abs(diff_total)>0])
    return mid_point


#CONVERSION

def conversion(nc_input_file,nc_output_file,nc_comp):
    #LOAD THE DATA:
    data = Dataset(nc_input_file)

    #CREATE THE OUTPUT FILE
    output = Dataset(nc_output_file,'w',format='NETCDF3_64BIT')
    output = replicate_netcdf_file(output,data)

    #CHECK THE SPECIFICATION OF JOINT DISTRIBUTION
    if 'joint_dist_vars' in data.ncattrs():
       dist_dims = getattr(data,'joint_dist_vars').split(',')
    else:
       raise IOError('variable name_of_bins must have a jd_dims attribute'.replace('name_of_bins',bins_name))

    #CREATE DIMENSIONS AND VARIABLES:
    for dims_id, dims in enumerate(dist_dims):
         dims_length=getattr(data,'jd_dist_'+dims)
	 output.createDimension(dims,dims_length)
         temp_dim = output.createVariable(dims,'d',(dims,))
	 temp_dim[:] = np.array(range(0,dims_length))+0.5

	 output.createDimension('s'+dims,dims_length+1)
         temp_dim = output.createVariable('s'+dims,'d',('s'+dims,))
	 temp_dim[:] = np.array(range(0,dims_length+1))
    output.sync()

    #DETERMINE WHICH VARIABLES TO DISTRIBUTE
    var_list = [ var for var in data.variables.keys() if 'joint_dist' in data.variables[var].ncattrs() ]
    for var in var_list:
	   #CREATE OUTPUT VAR:
	   print(var)
           jd_dir=var.split('_')[-1]

	   #FIND WHICH DIMENSIONS TO INCLUDE
	   var_dims = list(data.variables[var].dimensions)
	   if bins_name in var_dims:
	      var_dims.remove(bins_name)
	      for dims in dist_dims:
                if dims!=jd_dir:
	            var_dims.append(dims)
                else:
                    var_dims.append('s'+dims)

	   #CHECK IF THE NEW VAR DIMENSIONS EXIST. IF NOT, ADD THEM
	   for dims in var_dims:
	     if dims not in output.dimensions.keys():
	        output.createDimension(dims,len(data.dimensions[dims]))
		dim_var = output.createVariable(dims,'d',(dims,))
		dim_var[:] = data.variables[dims][:]
		output = replicate_netcdf_var(output,data,dims)

	   #CREATE OUTPUT VARIABLE WITH COMPRESSION
	   if var not in output.dimensions.keys():
	   #   #final_jd = output.createVariable(var,'f',tuple(var_dims),zlib=True)
                final_jd = output.createVariable(var,'d',tuple(var_dims),zlib=nc_comp)
                output = replicate_netcdf_var(output,data,var)
                index_staggered = list(output.variables[var].dimensions).index('s'+jd_dir)
                shape = list(final_jd.shape)
                shape[index_staggered]-=1
                shape_cat = list(final_jd.shape)
                shape_cat[index_staggered]=1
                temp = np.concatenate(
                                (np.reshape(data.variables[var],tuple(shape),order='F'),
                                np.zeros(shape_cat)),
                                axis=index_staggered
                                )
                #indices=zip(*[index[np.nonzero(temp)] for index in np.indices(temp.shape)])
                #print indices
                #print temp[np.nonzero(temp)]
                final_jd[:]=temp
                output.sync()
    output.close()

def collapse_dims_last(data,var,dist_dims):
        #RETRIEVE ONE OF THE VARIABLES
    	temp = data.variables[var]
	#if '_FillValue' in data.variables[var].ncattrs(): np.ma.masked_equal(temp,
	#                                                                     data.variables[var]._FillValue,
	#								     copy=False)
	temp_dims = data.variables[var].dimensions
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

def main():
    from optparse import OptionParser

    #Option parser
    parser = OptionParser()
    parser.add_option("-i","--input",dest="input_file",
                      help="Input file", metavar="FILE")
    parser.add_option("-o","--output",dest="output_file",
                      help="Output file", metavar="FILE")
    parser.add_option("-z","--zipped",dest="compression",
                      default=False, action="store_true",
                      help="Output file with NetCDF4 compression")
    parser.add_option("-a","--action",dest="action",
                      default="None",
                      help="Action to process. If unspecified, does nothing. Available actions: compute, conversion, pck")

    (options, args) = parser.parse_args()

    if options.action == "None":
        print('No action specified. Doing nothing')
    elif options.action == "compute":
        print('Computing joint distribution')
        joint_distribution_got(options.input_file,options.output_file,options.compression)
    elif options.action == "conversion":
        print('Converting bins to joint distribution')
        conversion(options.input_file,options.output_file,options.compression)
    else:
      raise IOError('Unknown action flag')
      
if __name__ == "__main__":
    main()
