subroutine radar_spectrum(&
    nbins,&		!in
    diameter_spec,&	!in
    back,&		!in
    back_spec,&		!in
    temp,&		!in
    press,&		!in
    frequency,&		!in
    particle_type,&	!in
    mass_size_a,&	!in
    mass_size_b,&	!in
    area_size_a,&	!in
    area_size_b,&	!in
    particle_spec)	!out

    ! this routine takes the backscattering spectrum depending on size and converts it
    ! into a spectrum depending on radar Doppler (=fall) velocity
    ! based on Spectra_simulator by P. Kollias
    ! 

    ! Current Code Owner: IGMK
    !
    ! History:
    !
    ! Version   Date       Comment
    ! -------   ----       -------
    ! 0.1       28/11/2012 converted from Matlab to Fortran - M. Maahn 
    ! 0.2       15/04/2013 Application of European Standards for Writing and
    !                      Documenting Exchangeable Fortran 90 Code - M. Maahn
    !
    ! Code Description:
    !   Language:		Fortran 90.
    !   Software Standards: "European Standards for Writing and  
    !     Documenting Exchangeable Fortran 90 Code". 
    !

    !in
    !nbins: No of bins + 1
    !diameter_spec: Diameter Spectrum (SI)
    !back_spec: backscattering cross section per volume in m²/m⁴ (includes number density)
    !temp: temperature in K
    !press: air pressure in Pa
    !frequency in GHz
    !particle_type: cloud|rain|snow|ice|hail|graupel
    !mass_size_a: a of mass size relation, needed for graupel, hail, snow, ice
    !mass_size_b: b of mass size relation, needed for graupel, hail, snow, ice
    !area_size_a: a of mass size relation, needed for graupel, hail, snow, ice
    !area_size_b: b of mass size relation, needed for graupel, hail, snow, ice
    !out
    !particle_spec particle spectrum in dependence of radar Dopple velocity in m6m-3/ms-1

    use kinds
    use settings
    use constants
    use report_module
    use dia2vel
    use rescale_spec
    implicit none

    integer,intent(in) ::  nbins !for Mie it is actually nbins+1
    real(kind=dbl), dimension(nbins),intent(in):: diameter_spec, back_spec
    character(5),intent(in) ::  particle_type
    real(kind=dbl), intent(in):: temp, frequency, press,back, mass_size_a, mass_size_b,&
    area_size_a, area_size_b

    real(kind=dbl), dimension(nbins):: vel_spec,dD_dU,back_vel_spec, back_spec_ref,&
    del_v_model, diameter_spec_cp
    real(kind=dbl), dimension(nbins) :: vel_spec_ext, back_vel_spec_ext
    real(kind=dbl), dimension(:,:), allocatable :: particle_spec_ext
    real(kind=dbl), intent(out), dimension(radar_nfft_aliased):: particle_spec
    real(kind=dbl), dimension(radar_nfft_aliased):: radar_velo_aliased, rho_particle
    real(kind=dbl):: del_v_radar, K2, dielec_water, wavelength, &
    delta_air, rho_air, rho, viscosity_air, my, del_d, Ze, K, &
    min_V_aliased, max_V_aliased
    integer :: ii, jj

!     integer(kind=long), intent(out) :: errorstatus
    integer(kind=long) :: errorstatus
    integer(kind=long) :: err = 0
    character(len=80) :: msg
    character(len=14) :: nameOfRoutine = 'radar_spectrum'


  

    if (verbose >= 2) call report(info,'Start of ', nameOfRoutine)

    ! get |K|**2 and lambda
    K2 = dielec_water(0.D0,temp-t_abs,frequency)
    wavelength = c / (frequency*1.d9)   ! m

    diameter_spec_cp(:) = diameter_spec(:)


    rho = rho_air(temp, press)
    my = viscosity_air(temp)/rho !kinematic viscosity

    !get the particle velocities depending on particle type!
    if (particle_type .eq. "cloud") then
	if (verbose >= 3) call report(info,'using: '//radar_fallVel_cloud//'to calculate fall velocity for cloud', nameOfRoutine)
        if (radar_fallVel_cloud .eq. "khvorostyanov01_spheres") then
            rho_particle(:) = rho_water
            call dia2vel_khvorostyanov01_spheres(err,nbins,diameter_spec_cp,rho,my,rho_particle,vel_spec)
        else if (radar_fallVel_cloud .eq. "khvorostyanov01_drops") then
            call dia2vel_khvorostyanov01_drops(err,nbins,diameter_spec_cp,rho,my,vel_spec)
        else if (radar_fallVel_cloud .eq. "rogers_drops") then
            call dia2vel_rogers_drops(err,nbins,diameter_spec_cp,rho,vel_spec)
        else if (radar_fallVel_cloud .eq. "pavlos_cloud") then
            call dia2vel_pavlos_cloud(err,nbins,diameter_spec_cp,vel_spec)
        else
            print*, "did not understand radar_fallVel_cloud ", radar_fallVel_cloud
            stop
        end if

    else if (particle_type .eq. "rain") then
	if (verbose >= 3) call report(info,'using: '//radar_fallVel_rain//'to calculate fall velocity for rain', nameOfRoutine)
        if (radar_fallVel_rain .eq. "khvorostyanov01_drops") then
            call dia2vel_khvorostyanov01_drops(err,nbins,diameter_spec_cp,rho,my,vel_spec)
        else if (radar_fallVel_rain .eq. "foote69_rain") then
            call dia2vel_foote69_rain(err,nbins,diameter_spec_cp,rho,temp,vel_spec)
        else if (radar_fallVel_rain .eq. "metek_rain") then
            call dia2vel_metek_rain(err,nbins,diameter_spec_cp,rho,temp,vel_spec)
        else if (radar_fallVel_rain .eq. "rogers_drops") then
            call dia2vel_rogers_drops(err,nbins,diameter_spec_cp,rho,vel_spec)
        else
            print*, "did not understand radar_fallVel_rain ", radar_fallVel_rain
            stop
        end if

    else if (particle_type .eq. "ice") then
	if (verbose >= 3) call report(info,'using: '//radar_fallVel_ice//'to calculate fall velocity for ice', nameOfRoutine)
        if (radar_fallVel_ice .eq. "heymsfield10_particles") then
            call dia2vel_heymsfield10_particles(err,nbins,diameter_spec_cp,rho,my,&
            mass_size_a,mass_size_b,area_size_a,area_size_b,vel_spec)
        else if (radar_fallVel_ice .eq. "khvorostyanov01_particles") then
            call dia2vel_khvorostyanov01_particles(err,nbins,diameter_spec_cp,rho,my,&
            mass_size_a,mass_size_b,area_size_a,area_size_b,vel_spec)
        else
            print*, "did not understand radar_fallVel_ice ", radar_fallVel_ice
            stop
        end if

    else if (particle_type .eq. "snow") then
	if (verbose >= 3) call report(info,'using: '//radar_fallVel_snow//'to calculate fall velocity for snow', nameOfRoutine)
        if (radar_fallVel_ice .eq. "heymsfield10_particles") then
            call dia2vel_heymsfield10_particles(err,nbins,diameter_spec_cp,rho,my,&
            mass_size_a,mass_size_b,area_size_a,area_size_b,vel_spec)
        else if (radar_fallVel_snow .eq. "khvorostyanov01_particles") then
            call dia2vel_khvorostyanov01_particles(err,nbins,diameter_spec_cp,rho,my,&
            mass_size_a,mass_size_b,area_size_a,area_size_b,vel_spec)
        else
            print*, "did not understand radar_fallVel_snow ", radar_fallVel_snow
            stop
        end if

    else if (particle_type .eq. "graup") then
	if (verbose >= 3) call report(info,'using: '//radar_fallVel_graupel//'to calculate fall velocity for graupel', nameOfRoutine)
        if (radar_fallVel_graupel .eq. "khvorostyanov01_spheres") then
            !reverse the mass-size relation to get the density assuming spheric shape, nonsense for rain and clouds
            rho_particle = mass_size_a * diameter_spec_cp**( mass_size_b-3.d0) * 6/pi
            call dia2vel_khvorostyanov01_spheres(err,nbins,diameter_spec_cp,rho,my,rho_particle,vel_spec)
        else if (radar_fallVel_graupel .eq. "rogers_graupel") then
            call dia2vel_rogers_graupel(err,nbins,diameter_spec_cp,vel_spec)
        else
            print*, "did not understand radar_fallVel_graupel ", radar_fallVel_graupel
            stop
        end if

    else if (particle_type .eq. "hail") then
	if (verbose >= 3) call report(info,'using: '//radar_fallVel_hail//'to calculate fall velocity for hail', nameOfRoutine)
        if (radar_fallVel_hail .eq. "khvorostyanov01_spheres") then
            !reverse the mass-size relation to get the density assuming spheric shape, nonsense for rain and clouds
            rho_particle = mass_size_a * diameter_spec_cp**( mass_size_b-3.d0) * 6/pi
            call dia2vel_khvorostyanov01_spheres(err,nbins,diameter_spec_cp,rho,my,rho_particle,vel_spec)
        else
            print*, "did not understand radar_fallVel_hail ", radar_fallVel_hail
            stop
        end if
    else
	errorstatus = fatal
	msg = particle_type//': Wrong particle type'
	call report(errorstatus, msg, nameOfRoutine)
	stop !return
    end if

    if (err /= 0) then
	msg = 'error in dia2vel_XX!'
	call report(err, msg, nameOfRoutine)
	errorstatus = err
	stop !return
    end if



    back_spec_ref = (1d0/ (K2*pi**5) ) * back_spec * (wavelength)**4 ![m⁶/m⁴]
    back_spec_ref =  back_spec_ref * 1d18 !now non-SI: [mm⁶/m³/m]


    del_d = (diameter_spec_cp(2)-diameter_spec_cp(1)) * 1.d3 ![mm]


    Ze = 1d18* (1d0/ (K2*pi**5) ) * back * (wavelength)**4


    !move from dimension to velocity!
    do jj=1,nbins-1
        dD_dU(jj) = (diameter_spec_cp(jj+1)-diameter_spec_cp(jj))/(vel_spec(jj+1)-vel_spec(jj)) ![m/(m/s)]
        !is all particles fall with the same velocity, dD_dU gets infinitive!
        if (abs(dD_dU(jj)) .ge. huge(dD_dU(jj))) then
            print*, jj,(diameter_spec_cp(jj+1)-diameter_spec_cp(jj)), (vel_spec(jj+1)-vel_spec(jj))
	    errorstatus = fatal
	    msg = "Stop in radar_spectrum: dD_dU is infinitive"
	    call report(errorstatus, msg, nameOfRoutine)
! 	    return
            stop
        end if
        if (verbose >= 4) print*,"jj,dD_dU(jj)",jj,dD_dU(jj)
        del_v_model(jj) = ABS(vel_spec(jj+1)-vel_spec(jj))



    end do
    dD_dU(nbins) = dD_dU(nbins-1)
    del_v_model(nbins) = del_v_model(nbins-1)
    back_vel_spec = back_spec_ref * ABS(dD_dU)  !non-SI: [mm⁶/m³/m * m/(m/s)]

    !get delta velocity
    del_v_radar = (radar_max_V-radar_min_V)/radar_nfft ![m/s]

    min_V_aliased = radar_min_V - radar_aliasing_nyquist_interv*(radar_max_V-radar_min_V)
    max_V_aliased = radar_max_V + radar_aliasing_nyquist_interv*(radar_max_V-radar_min_V)

    !create array from min_v to max_v iwth del_v_radar spacing -> velocity spectrum of radar
    radar_velo_aliased = (/(((ii*del_v_radar)+min_V_aliased),ii=0,radar_nfft_aliased)/) ! [m/s]


    !add vertical air motion to the observations
    if (radar_airmotion) then
	if (verbose >= 3) call report(info, "Averaging spectrum and Adding vertical air motion: "// radar_airmotion_model, nameOfRoutine)
        !constant air motion
        if (radar_airmotion_model .eq. "constant") then
            vel_spec = vel_spec + radar_airmotion_vmin
            !interpolate OR average (depending who's bins size is greater) from N(D) bins to radar bins.
            call rescale_spectra(err,nbins,radar_nfft_aliased,.false.,vel_spec,back_vel_spec,radar_velo_aliased,particle_spec) ! particle_spec in [mm⁶/m³/m * m/(m/s)]
        !step function
        else if (radar_airmotion_model .eq. "step") then

            allocate(particle_spec_ext(2,radar_nfft_aliased))
            !for vmin
            vel_spec_ext = vel_spec + radar_airmotion_vmin
            back_vel_spec_ext = back_vel_spec * radar_airmotion_step_vmin
            !interpolate OR average (depending who's bins size is greater) from N(D) bins to radar bins.
            call rescale_spectra(err,nbins,radar_nfft_aliased,.false.,vel_spec_ext,back_vel_spec_ext,radar_velo_aliased,&
            particle_spec_ext(1,:))! particle_spec in [mm⁶/m³/m * m/(m/s)]
            !for vmax
            vel_spec_ext = vel_spec + radar_airmotion_vmax
            back_vel_spec_ext = back_vel_spec *(1.d0-radar_airmotion_step_vmin)
            !interpolate OR average (depending who's bins size is greater) from N(D) bins to radar bins.
            call rescale_spectra(err,nbins,radar_nfft_aliased,.false.,vel_spec_ext,back_vel_spec_ext,radar_velo_aliased,&
            particle_spec_ext(2,:))
            !join results
            particle_spec = SUM(particle_spec_ext,1)
            deallocate(particle_spec_ext)
        !
        else if (radar_airmotion_model .eq. "linear") then
            allocate(particle_spec_ext(radar_airmotion_linear_steps,radar_nfft_aliased))
            delta_air = (radar_airmotion_vmax - radar_airmotion_vmin)/REAL(radar_airmotion_linear_steps -1)
            !     loop for linear steps
            do jj=1, radar_airmotion_linear_steps
                vel_spec_ext = vel_spec + radar_airmotion_vmin + (jj-1)*delta_air
                back_vel_spec_ext = back_vel_spec / REAL(radar_airmotion_linear_steps)
                !interpolate OR average (depending whos bins size is greater) from N(D) bins to radar bins.
                call rescale_spectra(err,nbins,radar_nfft_aliased,.false.,vel_spec_ext,back_vel_spec_ext,radar_velo_aliased,&
                particle_spec_ext(jj,:))
            end do
            !join results
            particle_spec = SUM(particle_spec_ext,1)
            deallocate(particle_spec_ext)
        else
	    errorstatus = fatal
	    msg = "unknown radar_airmotion_model: "// radar_airmotion_model
	    call report(errorstatus, msg, nameOfRoutine)
! 	    return
            stop
        end if
    else
        !no air motion, just rescale
	if (verbose >= 3) call report(info, "Averaging spectrum and Adding without vertical air motion", nameOfRoutine)
        call rescale_spectra(err,nbins,radar_nfft_aliased,.false.,vel_spec,back_vel_spec,radar_velo_aliased,particle_spec) ! particle_spec in [mm⁶/m³/m * m/(m/s)]
    end if

    if (err /= 0) then
	msg = 'error in rescale_spectra!'
	call report(err, msg, nameOfRoutine)
	errorstatus = err
	stop !return
    end if


    K = (Ze/SUM(particle_spec*del_v_radar))
    particle_spec = K* particle_spec

    if (verbose >= 4) print*,particle_type,"K",K
    if (verbose >= 4) print*,particle_type," Ze SUM(back_vel_spec)*del_v_model",10*log10(SUM(back_vel_spec*del_v_model))
    if (verbose >= 4) print*,particle_type," Ze SUM(back_vel_spec_ext)*del_v_model",10*log10(SUM(back_vel_spec_ext*del_v_model))
    if (verbose >= 4) print*,particle_type," Ze SUM(particle_spec)*del_v_radar",10*log10(SUM(particle_spec)*del_v_radar)



    errorstatus = err
    if (verbose >= 2) call report(info,'End of ', nameOfRoutine)

    return

end subroutine radar_spectrum
