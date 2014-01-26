import scipy.sparse.linalg as linalg
import scipy.fftpack as fftpack
import numpy as np
import copy
import spharm

class horizontal_vector_calculus:
    """
    This class computes derivatives in spherical coordinates
    assuming a C-grid.
    """
    def __init__(self,shape,lengths):
        self.lengths=lengths
        self.shape=shape

        self.Inv_Laplacian=(np.dstack(np.meshgrid(fftpack.fftfreq(self.shape[1],1.0/self.shape[1]),
                                       fftpack.fftfreq(self.shape[0],1.0/self.shape[0])))**2).sum(-1)
        self.Inv_Laplacian[self.Inv_Laplacian<1.0]=1.0
        self.Inv_Laplacian=1.0
        self.Inv_Laplacian/=self.lengths.area_lat_lon

        #Define the vector calculus in spherical harmonics:
        self.spharm=horizontal_vector_calculus_spherical_harmonics(self.shape,self.lengths)
        return

    def UV_from_Chi(self,Chi):
        U=np.empty(self.shape)
        U[:,1:]=(Chi[:,1:]-Chi[:,:-1])
        U[:,0]=(Chi[:,0]-Chi[:,-1])
        U/=self.lengths.zon_len_lat_slon
        #Make sure U is zero at the poles:
        U[0,:]=0.0
        U[-1,:]=0.0

        V=np.empty((self.shape[0]-1,self.shape[1]))
        V=(Chi[1:,:]-Chi[:-1,:])
        V/=self.lengths.mer_len_slat_lon
        return U, V

    def multiply_UV_by_lengths(self,U,V):
        return U*self.lengths.mer_len_lat_slon, V*self.lengths.zon_len_slat_lon

    def divide_UV_by_lengths(self,U,V):
        return U/self.lengths.mer_len_lat_slon, V/self.lengths.zon_len_slat_lon

    def DIV_from_UV(self,U,V):
        U,V = self.multiply_UV_by_lengths(U,V)
        DIV=np.empty(self.shape)
        DIV[:,:-1]=(U[:,1:]-U[:,:-1])
        DIV[:,-1]=(U[:,0]-U[:,-1])

        DIV[0,:]+=V[0,:].mean()
        DIV[1:-1,:]+=V[1:,:]-V[:-1,:]
        DIV[-1,:]+=-V[-1,:].mean()
        return DIV

    def helmholtz(self,Chi,wavenumber_sqr=0):
        """
        This is the positive definite operator $-nabla^2+kappa^2$
        """
        U, V = self.UV_from_Chi(Chi)
        DIV = self.DIV_from_UV(U,V)
        DIV*=-1.0
        DIV+=wavenumber_sqr*Chi*self.lengths.area_lat_lon
        return DIV

    def inverse_helmholtz_approx(self,DIV,wavenumber_sqr=0):
        """
        Approximate inverse to the helmholtz equation on the sphere.
        Uses fourier transforms. Not used in function inverse_helmholtz.
        This is the positive definite operator $-nabla^2+kappa^2$
        """
        return fftpack.ifft2(fftpack.fft2(DIV/self.lengths.area_lat_lon)/(self.Inv_Laplacian+wavenumber_sqr)).real

    def inverse_helmholtz(self,DIV,wavenumber_sqr=0):
        """
        Finds the inverse to the helmholtz equation on the sphere using
        the conjugate gradient iteration.
        Uses the spherical harmonic solution as an educated guess.
        This is the positive definite operator $-nabla^2+kappa^2$
        """
        DIV[0,:]=DIV[0,:].mean(-1)
        DIV[-1,:]=DIV[-1,:].mean(-1)
        DIV-=DIV.mean()
        div=self._flatten_spherical_array(DIV)

        if wavenumber_sqr>=0:
            helmholtz=linalg.LinearOperator((len(div),len(div)),
                                        (lambda x: self._flatten_spherical_array(
                                                        self.helmholtz(
                                                            self._unflatten_spherical_array(x,DIV[1:-1,:].shape),wavenumber_sqr=wavenumber_sqr
                                                                                    )
                                                                            )),
                                        dtype=np.float)
            initial_value=linalg.LinearOperator((len(div),len(div)),
                                        (lambda x: self._flatten_spherical_array(
                                                        self.spharm.inverse_helmholtz(
                                                                        self._unflatten_spherical_array(x,DIV[1:-1,:].shape),
                                                                        wavenumber_sqr=wavenumber_sqr
                                                                                                    )
                                                                          )
                                                    ),
                                        dtype=np.float)

            velopot=initial_value.matvec(div)
            sol ,info=linalg.cg(helmholtz,div,x0=velopot,maxiter=20)
            #sol=velopot
        else:
            raise ValueError('The wavenumer_sqr argument should be nonnegative')

        return self._unflatten_spherical_array(sol,DIV[1:-1,:].shape)

    def _flatten_spherical_array(self,array): 
            return np.hstack(([array[0,0]],array[1:-1,:].flatten(),[array[-1,-1]]))

    def _unflatten_spherical_array(self,array,shape): 
            array_out=np.concatenate(
                        (array[0]*np.ones((1,shape[1])),
                         np.reshape(array[1:-1],shape),
                         array[-1]*np.ones((1,shape[1]))),
                         axis=0
                         )
            return array_out

class horizontal_vector_calculus_spherical_harmonics:
    """
    This class computes derivatives in spherical coordinates
    assuming a regular grid with values at the cell center of a C-grid.
    """
    def __init__(self,shape,lengths):
        self.shape=shape
        self.lengths=lengths
        self.spharm=spharm.Spharmt(self.shape[1],self.shape[0],legfunc='computed')

        #Define the inverse laplacian in spectral space:
        setattr(self,'_s_Inv_Laplacian',self.spharm.grdtospec(np.zeros(self.shape)))
        self.legendre_order=spharm.getspecindx(self.shape[0]-1)[1]
        self._s_Inv_Laplacian[1:]=(self.legendre_order[1:]*(self.legendre_order[1:]+1))
        self._s_Inv_Laplacian[0]=1.0
        self._s_Inv_Laplacian/=self.lengths.rsphere**2

        #To shift input to have first index at Greenwich meridian
        self.shift_index=np.argmin(np.abs(self.lengths.lon))

        #For putting on the C-grid:
        #self.spharm_regrid=spharm.Spharmt(self.shape[1]*2,(self.shape[0]-1)*2+1,legfunc='computed')
        return

    def UV_from_Chi(self,g_Chi):
        s_Chi=self.spharm.grdtospec(g_Chi)
        g_U, g_V = self.spharm.getgrad(s_Chi)
        return g_U, g_V

    def USVS_from_Chi(self,Chi):
        s_Chi=self.spharm.grdtospec(g_Chi)
        g_U, g_V = self.spharm.getgrad(s_Chi)

        slice_lon=slice(1,self.spharm_regrid.nlon+1,2)
        slice_lat=slice(0,self.spharm_regrid.nlat+1,2)
        shift_offset=1
        US = np.flipud(np.roll(
                          spharm.regrid(self.spharm,self.spharm_regrid,g_U)[slice_lat,slice_lon],
                  self.shift_index+shift_offset,axis=1))
        slice_lon=slice(0,self.spharm_regrid.nlon+1,2)
        slice_lat=slice(1,self.spharm_regrid.nlat,2)
        shift_offset=0
        VS = np.flipud(np.roll(
                          spharm.regrid(self.spharm,self.spharm_regrid,g_V)[slice_lat,slice_lon],
                  self.shift_index+shift_offset,axis=1))
        return US, VS

    def DIV_from_UV(self,g_U,g_V):
        s_VOR, s_DIV = self.spharm.getvrtdivspec(g_U,g_V)
        g_DIV=self.spharm.spectogrd(s_DIV)
        return g_DIV

    def helmholtz(self,Chi,wavenumber_sqr=0):
        """
        This is the positive definite operator $-nabla^2+kappa^2$
        """
        U, V = self.UV_from_Chi(Chi)
        DIV = self.DIV_from_UV(U,V)
        DIV*=lengths.area_lat_lon
        DIV*=-1.0
        DIV+=wavenumber_sqr*Chi
        return DIV

    def UV_from_DIV_no_VOR(self,g_DIV):
        s_DIV=self.spharm.grdtospec(g_DIV)
        s_VOR=self.spharm.grdtospec(np.zeros_like(g_DIV))
        g_U, g_V = self.spharm.getuv(s_VOR,s_DIV)
        return g_U, g_V

    def Chi_from_UV(self,g_U,g_V):
        g_Psi, g_Chi=self.spharm.getpsichi(g_U,g_V)
        return g_Chi

    #def inverse_helmholtz_slow(self,DIV,wavenumber_sqr=0):
    #    """
    #    Find the inverse of the helmholtz equation on the sphere using
    #    spherical harmonics. Slow but agnostic implementation.
    #    This is the positive definite operator $-nabla^2+kappa^2$
    #    """
    #    DIV_per_area=DIV/self.lengths.area_lat_lon
    #
    #    U, V =self.UV_from_DIV_no_VOR(DIV_per_area)
    #    Chi = -self.Chi_from_UV(U,V)
    #    return Chi

    def inverse_helmholtz(self,DIV,wavenumber_sqr=0):
        """
        Find the inverse of the helmholtz equation on the sphere using
        spherical harmonics. Fast implementation.
        This is the positive definite operator $-nabla^2+kappa^2$
        """
        s_Chi=self.spharm.grdtospec(DIV/self.lengths.area_lat_lon)/(self._s_Inv_Laplacian+wavenumber_sqr)
        s_Chi[0]=0.0
        return self.spharm.spectogrd(s_Chi)
        
class vertical_discrete_calculus:
    def __init__(self,shape):
        self.shape=shape
        self._s_Inv_Laplacian=4*np.sin((np.pi*np.array(range(0,self.shape)))/(2*self.shape))**2
        return

    def W_from_Chi(self,Chi):
        W=np.empty(len(Chi)+1)
        W[1:-1]=(Chi[1:]-Chi[:-1])
        W[0]=0.0
        W[-1]=0.0
        return W

    def DIV_from_W(self,W):
        DIV=np.empty(len(W)-1)
        DIV=W[1:]-W[:-1]
        return DIV

    def laplacian(self,Chi):
        """
        This is the positive definite operator $-nabla^2$
        """
        W = self.W_from_Chi(Chi)
        DIV = DIV_from_W(W)
        DIV *= -1.0
        return DIV

def retrieve_coordinates(data,pair):
    lat=getattr(data,pair['latitude'][:])
    lat=np.reshape(lat,(len(lat),1))

    lon=getattr(data,pair['longitude'][:])
    lon=np.reshape(lon,(1,len(lon)))
    return lat,lon

def conjugate_dimension(pair,dim_id):
    pair_tmp=copy.copy(pair)
    if pair_tmp[dim_id][0]=='s':
        pair_tmp[dim_id]=pair_tmp[dim_id][1:]
    else:
        pair_tmp[dim_id]='s'+pair_tmp[dim_id]
    return pair_tmp

class coords:
    def __init__(self,data):
        self.rsphere=6.3712e6
        for coordinate in ['lat','lon','slon','slat']:
            setattr(self,coordinate,data.variables[coordinate][:])

        for pair in [{'latitude':'lat','longitude':'slon'},{'latitude':'slat','longitude':'lon'}]:

            lat, lon = retrieve_coordinates(self,conjugate_dimension(pair,'longitude'))
            lz='zon_len_'+'_'.join([pair[key] for key in pair.keys()])
            setattr(self,lz,np.zeros_like(lon))
            getattr(self,lz)[:,:-1]=lon[:,1:]-lon[:,:-1]
            getattr(self,lz)[:,-1]=lon[:,0] + 360.0 -lon[:,-1]
            setattr(self,lz,getattr(self,lz)[...]*self.rsphere*np.pi/180.0*np.cos(lat*np.pi/180.0))

            lat, lon = retrieve_coordinates(self,conjugate_dimension(pair,'latitude'))
            if pair['latitude']=='slat':
                lat=lat[1:-1]
            lm='mer_len_'+'_'.join([pair[key] for key in pair.keys()])
            setattr(self,lm,np.zeros((lat.shape[0]+1,lat.shape[1])))
            getattr(self,lm)[0,:]=lat[0,:]- (-90.0)
            getattr(self,lm)[1:-1,:]=lat[1:,:]-lat[:-1,:]
            getattr(self,lm)[-1,:]=90.0 - lat[-1,:]
            getattr(self,lm)[...]*=self.rsphere*np.pi/180.0


        lat, lon = retrieve_coordinates(self,{'latitude':'slat','longitude':'slon'})
        dsin_lat = np.zeros((lat.shape[0]+1,lat.shape[1]))
        dsin_lat[0,:] = np.sin(lat[0,:]*np.pi/180.0) - (-1.0)
        dsin_lat[1:-1,:] = np.sin(lat[1:,:]*np.pi/180.0)-np.sin(lat[:-1,:]*np.pi/180.0)
        dsin_lat[-1,:] = 1.0 - np.sin(lat[-1,:]*np.pi/180.0)
        dlon = np.zeros_like(lon)
        dlon[:,:-1] = lon[:,1:]-lon[:,:-1]
        dlon[:,-1] = lon[:,0]+ 360.0-lon[:,-1]
        dlon*=np.pi/180.0

        darea=self.rsphere**2*dsin_lat*dlon
        darea[0,:]=darea[0,:].mean()
        darea[-1,:]=darea[-1,:].mean()

        area='area_lat_lon'
        setattr(self,area,darea)

        return

class vector_calculus_spherical:
    def __init__(self,shape,lengths):
        self.lengths=lengths
        self.shape=shape
        self.spherical_calculus=horizontal_vector_calculus(self.shape[1:],self.lengths)
        self.spherical_calculus_spharm=horizontal_vector_calculus_spherical_harmonics(self.shape[1:],self.lengths)
        self.vertical_calculus=vertical_discrete_calculus(self.shape[0])
        return

    def add_first_axis(self,array):
        return np.reshape(array,(1,)+array.shape)

    def inverse_laplacian_approx(self,DIV):
        """
        An approximate inverse to the laplacian in spherical coordinates.
        Can be used as an initial guess for an iteration.
        Inverts the positive definite $-nabla^2$ operator. 
        """
        Chi = fftpack.idct(
                np.concatenate(
                             map(
                                lambda x: self.add_first_axis(
                                            self.spherical_calculus.inverse_helmholtz(np.squeeze(x[0]),x[1])
                                            ),
                                zip(
                                    np.split(fftpack.dct(DIV,axis=0,norm='ortho',overwrite_x=True),DIV.shape[0],axis=0),
                                    self.vertical_calculus._s_Inv_Laplacian
                                    )
                                )
                             ),
                        axis=0,norm='ortho',overwrite_x=True)
        return Chi

    def UVW_mass_from_Chi(self,Chi):
        """
        Returns the mass fluxes across boundaries
        """
        U, V, W = self.UVW_from_Chi(Chi)

        U*=np.reshape(self.lengths.mer_len_lat_slon,(1,)+self.lengths.mer_len_lat_slon.shape)
        V*=np.reshape(self.lengths.zon_len_slat_lon,(1,)+self.lengths.zon_len_slat_lon.shape)
        W*=np.reshape(self.lengths.area_lat_lon,(1,)+self.lengths.area_lat_lon.shape)
        return U, V, W

    def UVW_from_Chi(self,Chi):
        U = np.concatenate(
                             map(
                                lambda x: self.add_first_axis(
                                            self.spherical_calculus.UV_from_Chi(np.squeeze(x))[0]
                                            ),
                                    np.split(Chi,Chi.shape[0],axis=0)
                                )
                             )
        V = np.concatenate(
                             map(
                                lambda x: self.add_first_axis(
                                            self.spherical_calculus.UV_from_Chi(np.squeeze(x))[1]
                                            ),
                                    np.split(Chi,Chi.shape[0],axis=0)
                                )
                             )
        W = np.apply_along_axis(lambda x: self.vertical_calculus.W_from_Chi(x),0,Chi)
        return U, V, W

    def DIV_from_UVW_mass(self,Um,Vm,Wm):
        """
        Returns the mass divergence from mass fluxes across boundaries
        """
        DIV = self.DIV_from_UVW(
                                Um/np.reshape(self.lengths.mer_len_lat_slon,(1,)+self.lengths.mer_len_lat_slon.shape),
                                Vm/np.reshape(self.lengths.zon_len_slat_lon,(1,)+self.lengths.zon_len_slat_lon.shape),
                                Wm/np.reshape(self.lengths.area_lat_lon,(1,)+self.lengths.area_lat_lon.shape)
                                )
        return DIV

    def DIV_from_UVW_spharm(self,U,V,W):
        DIV = np.concatenate(
                             map(
                                lambda x: self.add_first_axis(
                                            self.spherical_calculus_spharm.DIV_from_UV(np.squeeze(x[0]),np.squeeze(x[1]))
                                            ),
                                    zip(np.split(U,U.shape[0],axis=0),
                                        np.split(V,V.shape[0],axis=0)
                                        )
                                )
                             )
        DIV+=np.apply_along_axis(lambda x: self.vertical_calculus.DIV_from_W(x),0,W)*self.add_first_axis(self.lengths.area_lat_lon)
        return DIV

    def DIV_from_UVW(self,U,V,W):
        DIV = np.concatenate(
                             map(
                                lambda x: self.add_first_axis(
                                            self.spherical_calculus.DIV_from_UV(np.squeeze(x[0]),np.squeeze(x[1]))
                                            ),
                                    zip(np.split(U,U.shape[0],axis=0),
                                        np.split(V,V.shape[0],axis=0)
                                        )
                                )
                             )
        DIV+=np.apply_along_axis(lambda x: self.vertical_calculus.DIV_from_W(x),0,W)*self.add_first_axis(self.lengths.area_lat_lon)
        return DIV

    def laplacian(self,Chi):
        """
        This is the positive definite $-nabla^2$ operator. 
        """
        U, V, W = self.UVW_from_Chi(Chi)
        DIV = self.DIV_from_UVW(U,V,W)
        DIV *= -1.0
        return DIV

    def inverse_laplacian(self,DIV,maxiter=10):
        """
        Finds the inverse to the laplacian equation on the sphere using
        the conjugate gradient iteration.
        Uses the spherical harmonic solution as an educated guess.
        Inverts the positive definite $-nabla^2$ operator. 

        """
        DIV[:,0,:]=np.mean(DIV[:,0,:],axis=-1,keepdims=True)
        DIV[:,-1,:]=np.mean(DIV[:,-1,:],axis=-1,keepdims=True)
        DIV-=DIV.mean()
        div=self._flatten_spherical_array(DIV)

        laplacian=linalg.LinearOperator((len(div),len(div)),
                                    (lambda x: self._flatten_spherical_array(
                                                    self.laplacian(
                                                        self._unflatten_spherical_array(x,DIV[:,1:-1,:].shape)
                                                                                )
                                                                        )),
                                    dtype=np.float)
        initial_value=linalg.LinearOperator((len(div),len(div)),
                                    (lambda x: self._flatten_spherical_array(
                                                    self.inverse_laplacian_approx(
                                                                    self._unflatten_spherical_array(x,DIV[:,1:-1,:].shape)
                                                                                                )
                                                                      )
                                                ),
                                    dtype=np.float)

        velopot=initial_value.matvec(div)
        if maxiter==0:
            sol=velopot
        else:
            sol ,info=linalg.cg(laplacian,div,x0=velopot,maxiter=maxiter)

        return self._unflatten_spherical_array(sol,DIV[:,1:-1,:].shape)

    def _flatten_spherical_array(self,array): 
            return np.hstack((array[:,0,0],array[:,1:-1,:].flatten(),array[:,-1,-1]))

    def _unflatten_spherical_array(self,array,shape): 
            array_out=np.concatenate(
                         (
                         np.reshape(array[:shape[0]],(shape[0],1,1))*np.ones((1,1,shape[2])),
                         np.reshape(array[shape[0]:-shape[0]],shape),
                         np.reshape(array[-shape[0]:],(shape[0],1,1))*np.ones((1,1,shape[2]))
                         ),
                         axis=1)
            return array_out
