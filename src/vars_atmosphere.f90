module vars_atmosphere

  use kinds
  use settings, only: noutlevels,nfrq,NSTOKES,nummu
  use report_module

  implicit none
  save

  real(kind=sgl) :: cwp,iwp,rwp,swp,gwp,hwp

  character(len=3) :: atmo_input_type

  integer(kind=long) :: atmo_ngridx, atmo_ngridy
  character(len=2), allocatable, dimension(:,:) :: atmo_month, atmo_day
  character(len=4), allocatable, dimension(:,:) :: atmo_year, atmo_time
!   real(kind=sgl), allocatable, dimension(:,:) :: atmo_deltax, atmo_deltay
  integer(kind=long), allocatable, dimension(:,:) :: atmo_model_i, atmo_model_j
  real(kind=sgl), allocatable, dimension(:,:) :: atmo_lon,atmo_lat,atmo_wind10u,atmo_wind10v
  real(kind=sgl), allocatable, dimension(:,:,:) :: atmo_obs_height
  !number of layer can vary (from python)
  integer(kind=long), allocatable, dimension(:,:) :: atmo_nlyrs, atmo_unixtime
  integer(kind=long) :: atmo_max_nlyrs

  real(kind=dbl), allocatable, dimension(:,:) :: atmo_groundtemp, atmo_iwv

  integer(kind=long), allocatable, dimension(:,:) :: sfc_type, sfc_model
  real(kind=dbl), allocatable, dimension(:,:) :: sfc_salinity, sfc_slf, sfc_sif
  character(len=1), allocatable, dimension(:,:) ::  sfc_refl
  real(kind=dbl), allocatable, dimension(:,:,:,:,:) :: sfc_emissivity

  real(kind=dbl), allocatable, dimension(:,:,:) :: atmo_relhum_lev,&
       atmo_press_lev, &
       atmo_temp_lev, &
       atmo_hgt_lev

  real(kind=dbl), allocatable, dimension(:,:,:) :: atmo_relhum,&
       atmo_press, &
       atmo_temp, &
       atmo_hgt, &
       atmo_delta_hgt_lev, &
       atmo_airturb, &
       atmo_wind_uv, &
       atmo_turb_edr, &
       atmo_vapor_pressure, &
       atmo_rho_vap, &
       atmo_q_hum, &
       atmo_wind_w

  real(kind=dbl), allocatable, dimension(:,:,:,:) :: atmo_hydro_q,&
       atmo_hydro_reff, &
       atmo_hydro_n

  real(kind=dbl), allocatable, dimension(:,:,:) :: atmo_hydro_q_column,&
       atmo_hydro_reff_column, &
       atmo_hydro_n_column

  !trashbin for time dependent radar properties, first col is radarnoise_factor, second currently not used
  real(kind=dbl), allocatable, dimension(:,:,:) :: atmo_radar_prop
  contains
!##################################################################################################################################
  subroutine screen_input(errorstatus)

    use settings, only: verbose, input_file, input_pathfile, &
      output_path, nc_out_file, file_desc, freq_str
    use descriptor_file, only: moment_in_arr, n_hydro

    implicit none

    integer(kind=long), intent(out) :: errorstatus
    integer(kind=long) :: err = 0
    character(len=80) :: msg
    character(len=14) :: nameOfRoutine = 'screen_input'

! work variables
    real(kind=dbl), dimension(8) :: dum_8

    if (verbose >= 3) call report(info,'Start of ', nameOfRoutine)


    atmo_input_type = trim(input_pathfile(len_trim(input_pathfile)-2:len_trim(input_pathfile)))

    if ((atmo_input_type .ne. 'cla') .and. (atmo_input_type .ne. 'lev') .and. (atmo_input_type .ne. 'lay'))  then
        msg = "Unknown ascii input file type"//trim(atmo_input_type)
        call report(err,msg,nameOfRoutine)
        errorstatus = fatal
        return
    end if

! OPEN input file
    open(UNIT=14, FILE=TRIM(input_pathfile),&
    STATUS='OLD', iostat=err)
    if (err /= 0) then
        msg = "Read error: Cannot open file "//TRIM(input_pathfile)
        call report(err,msg,nameOfRoutine)
        errorstatus = fatal
        return
    end if

! Screen NEW input file format
    if ((atmo_input_type == 'lev') .or. (atmo_input_type == 'lay')) then
! READ atmo_ngridx, atmo_ngridy, atmo_max_nlyrs
      read(14,*) atmo_ngridx, atmo_ngridy, atmo_max_nlyrs
! add layers/levels to be able to insert the output levels as a new layers/levels
      atmo_max_nlyrs = atmo_max_nlyrs + noutlevels
      close(14)
    end if

! Screen OLD input file format
    if (atmo_input_type == 'cla') then
! consistency CHECK
      call assert_true(err,(n_hydro==5),&
          "STOP: 'classic' input with n_hydro /= 5")
      call assert_true(err,((maxval(moment_in_arr) == 3).and.(minval(moment_in_arr) == 3) ),&
          "STOP: 'classic' input with moment_in /= 3")
      if (err > 0) then
        errorstatus = fatal
        msg = "assertation error"
        call report(errorstatus, msg, nameOfRoutine)
        return
      end if
      read(14,*) dum_8
      atmo_ngridx = dum_8(5)
      atmo_ngridy = dum_8(6)
      atmo_max_nlyrs = dum_8(7)
! add layers/levels to be able to insert the output levels as a new layers/levels
      atmo_max_nlyrs = atmo_max_nlyrs + noutlevels
      close(14)
    endif

    nc_out_file = trim(output_path)//"/"//trim(input_file(1:len_trim(input_file)-4))//&
    trim(freq_str)//trim(file_desc)//'.nc'

    errorstatus = err
    if (verbose >= 3) call report(info,'End of ', nameOfRoutine)
    return

  end subroutine screen_input
!##################################################################################################################################
  subroutine allocate_atmosphere_vars(errorstatus)

    use descriptor_file, only: n_hydro
    use nan, only: nan_dbl

    integer(kind=long), intent(out) :: errorstatus
    integer(kind=long) :: err
    character(len=80) :: msg
    character(len=14) :: nameOfRoutine = 'allocate_atmosphere_vars'

    err = 0

    if (verbose >= 3) call report(info,'Start of ', nameOfRoutine)

    call assert_true(err,(atmo_ngridx>0),&
        "atmo_ngridx must be greater zero")
    call assert_true(err,(atmo_ngridy>0),&
        "atmo_ngridy must be greater zero")
    call assert_true(err,(atmo_max_nlyrs>0),&
        "atmo_max_nlyrs must be greater zero")
    call assert_true(err,(noutlevels>0),&
        "noutlevels must be greater zero")
    if (err > 0) then
        errorstatus = fatal
        msg = "assertation error"
        call report(errorstatus, msg, nameOfRoutine)
        return
    end if

    allocate(atmo_month(atmo_ngridx,atmo_ngridy))
    allocate(atmo_day(atmo_ngridx,atmo_ngridy))
    allocate(atmo_year(atmo_ngridx,atmo_ngridy))
    allocate(atmo_time(atmo_ngridx,atmo_ngridy))
!     allocate(atmo_deltax(atmo_ngridx,atmo_ngridy))
!     allocate(atmo_deltay(atmo_ngridx,atmo_ngridy))
    allocate(atmo_model_i(atmo_ngridx,atmo_ngridy))
    allocate(atmo_model_j(atmo_ngridx,atmo_ngridy))
    allocate(atmo_lon(atmo_ngridx,atmo_ngridy))
    allocate(atmo_lat(atmo_ngridx,atmo_ngridy))
    allocate(atmo_wind10u(atmo_ngridx,atmo_ngridy))
    allocate(atmo_wind10v(atmo_ngridx,atmo_ngridy))
    allocate(atmo_obs_height(atmo_ngridx,atmo_ngridy,noutlevels))

    !todo: add assert ngrid_X > 0 etc etc
    allocate(atmo_nlyrs(atmo_ngridx,atmo_ngridy))
    allocate(atmo_unixtime(atmo_ngridx,atmo_ngridy))
    allocate(atmo_groundtemp(atmo_ngridx,atmo_ngridy))
    allocate(atmo_iwv(atmo_ngridx,atmo_ngridy))

    ! surface variables
    allocate(sfc_type(atmo_ngridx,atmo_ngridy))
    allocate(sfc_model(atmo_ngridx,atmo_ngridy))
    allocate(sfc_refl(atmo_ngridx,atmo_ngridy))
    allocate(sfc_salinity(atmo_ngridx,atmo_ngridy))
    allocate(sfc_slf(atmo_ngridx,atmo_ngridy))
    allocate(sfc_sif(atmo_ngridx,atmo_ngridy))
    allocate(sfc_emissivity(atmo_ngridx,atmo_ngridy,nfrq,NSTOKES,nummu))

    allocate(atmo_relhum_lev(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs+1))
    allocate(atmo_press_lev(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs+1))
    allocate(atmo_temp_lev(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs+1))
    allocate(atmo_hgt_lev(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs+1))

    allocate(atmo_relhum(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs))
    allocate(atmo_press(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs))
    allocate(atmo_temp(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs))

    allocate(atmo_vapor_pressure(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs))
    allocate(atmo_rho_vap(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs))
    allocate(atmo_q_hum(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs))

    allocate(atmo_hgt(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs))
    allocate(atmo_delta_hgt_lev(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs))
    allocate(atmo_airturb(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs))
    allocate(atmo_wind_w(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs))
    allocate(atmo_wind_uv(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs))
    allocate(atmo_turb_edr(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs))

    allocate(atmo_hydro_q(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs,n_hydro))
    allocate(atmo_hydro_q_column(atmo_ngridx,atmo_ngridy,n_hydro))
    allocate(atmo_hydro_reff(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs,n_hydro))
    allocate(atmo_hydro_reff_column(atmo_ngridx,atmo_ngridy,n_hydro))
    allocate(atmo_hydro_n(atmo_ngridx,atmo_ngridy,atmo_max_nlyrs,n_hydro))
    allocate(atmo_hydro_n_column(atmo_ngridx,atmo_ngridy,n_hydro))

    allocate(atmo_radar_prop(atmo_ngridx,atmo_ngridy,2))


    atmo_month(:,:) = "na"
    atmo_day(:,:) = "na"
    atmo_year(:,:) = "na"
    atmo_time(:,:) = "na"
!     atmo_deltax(:,:) = nan_dbl()
!     atmo_deltay(:,:) = nan_dbl()
    atmo_model_i(:,:) = -9999
    atmo_model_j(:,:) = -9999
    atmo_lon(:,:) = nan_dbl()
    atmo_lat(:,:) = nan_dbl()
    atmo_wind10u(:,:) = nan_dbl()
    atmo_wind10v(:,:) = nan_dbl()
    atmo_obs_height(:,:,:) = nan_dbl()

    atmo_relhum_lev(:,:,:) = nan_dbl()
    atmo_press_lev(:,:,:) = nan_dbl()
    atmo_temp_lev(:,:,:) = nan_dbl()
    atmo_hgt_lev(:,:,:) = nan_dbl()

    atmo_groundtemp(:,:) = nan_dbl()
    atmo_iwv(:,:) = nan_dbl()

    sfc_type(:,:) = -9999
    sfc_model(:,:) = -9999
    sfc_refl(:,:) = "S"
    sfc_salinity(:,:) = nan_dbl()
    sfc_slf(:,:) = nan_dbl()
    sfc_sif(:,:) = nan_dbl()

    atmo_relhum(:,:,:) = nan_dbl()
    atmo_press(:,:,:) = nan_dbl()
    atmo_temp(:,:,:) = nan_dbl()
    atmo_vapor_pressure(:,:,:) = nan_dbl()
    atmo_rho_vap(:,:,:) = nan_dbl()
    atmo_q_hum(:,:,:) = nan_dbl()
    atmo_hgt(:,:,:) = nan_dbl()
    atmo_delta_hgt_lev(:,:,:) = nan_dbl()
    atmo_airturb(:,:,:) = nan_dbl()
    atmo_wind_uv(:,:,:) = nan_dbl()
    atmo_turb_edr(:,:,:) = nan_dbl()
    atmo_wind_w(:,:,:) =nan_dbl()

    atmo_hydro_q(:,:,:,:) = nan_dbl()
    atmo_hydro_q_column(:,:,:) = nan_dbl()
    atmo_hydro_reff(:,:,:,:) = nan_dbl()
    atmo_hydro_reff_column(:,:,:) = nan_dbl()
    atmo_hydro_n(:,:,:,:) = nan_dbl()
    atmo_hydro_n_column(:,:,:) = nan_dbl()

    atmo_nlyrs(:,:) = -9999
    atmo_unixtime(:,:) = -9999

    atmo_radar_prop(:,:,:) = nan_dbl()

    errorstatus = err
    if (verbose >= 3) call report(info,'End of ', nameOfRoutine)

    return

  end subroutine allocate_atmosphere_vars
!##################################################################################################################################
  subroutine deallocate_atmosphere_vars()

    if (allocated(atmo_nlyrs)) deallocate(atmo_nlyrs)
    if (allocated(atmo_unixtime)) deallocate(atmo_unixtime)

    if (allocated(atmo_relhum_lev)) deallocate(atmo_relhum_lev)
    if (allocated(atmo_press_lev)) deallocate(atmo_press_lev)
    if (allocated(atmo_temp_lev)) deallocate(atmo_temp_lev)
    if (allocated(atmo_hgt_lev)) deallocate(atmo_hgt_lev)

    if (allocated(atmo_relhum)) deallocate(atmo_relhum)
    if (allocated(atmo_press)) deallocate(atmo_press)
    if (allocated(atmo_vapor_pressure)) deallocate(atmo_vapor_pressure)
    if (allocated(atmo_rho_vap)) deallocate(atmo_rho_vap)
    if (allocated(atmo_q_hum)) deallocate(atmo_q_hum)
    if (allocated(atmo_temp)) deallocate(atmo_temp)
    if (allocated(atmo_hgt)) deallocate(atmo_hgt)
    if (allocated(atmo_delta_hgt_lev)) deallocate(atmo_delta_hgt_lev)
    if (allocated(atmo_airturb)) deallocate(atmo_airturb)
    if (allocated(atmo_wind_w)) deallocate(atmo_wind_w)
    if (allocated(atmo_wind_uv)) deallocate(atmo_wind_uv)
    if (allocated(atmo_turb_edr)) deallocate(atmo_turb_edr)

    if (allocated(atmo_hydro_q)) deallocate(atmo_hydro_q)
    if (allocated(atmo_hydro_q_column)) deallocate(atmo_hydro_q_column)
    if (allocated(atmo_hydro_reff)) deallocate(atmo_hydro_reff)
    if (allocated(atmo_hydro_reff_column)) deallocate(atmo_hydro_reff_column)
    if (allocated(atmo_hydro_n)) deallocate(atmo_hydro_n)
    if (allocated(atmo_hydro_n_column)) deallocate(atmo_hydro_n_column)

    if (allocated(atmo_groundtemp)) deallocate(atmo_groundtemp)
    if (allocated(atmo_iwv)) deallocate(atmo_iwv)
    if (allocated(sfc_type)) deallocate(sfc_type)
    if (allocated(sfc_model)) deallocate(sfc_model)
    if (allocated(sfc_refl)) deallocate(sfc_refl)
    if (allocated(sfc_salinity)) deallocate(sfc_salinity)
    if (allocated(sfc_slf)) deallocate(sfc_slf)
    if (allocated(sfc_sif)) deallocate(sfc_sif)
    if (allocated(sfc_emissivity)) deallocate(sfc_emissivity)

    if (allocated(atmo_month)) deallocate(atmo_month)
    if (allocated(atmo_day)) deallocate(atmo_day)
    if (allocated(atmo_year)) deallocate(atmo_year)
    if (allocated(atmo_time)) deallocate(atmo_time)
!     if (allocated(atmo_deltax)) deallocate(atmo_deltax)
!     if (allocated(atmo_deltay)) deallocate(atmo_deltay)
    if (allocated(atmo_model_i)) deallocate(atmo_model_i)
    if (allocated(atmo_model_j)) deallocate(atmo_model_j)
    if (allocated(atmo_lon)) deallocate(atmo_lon)
    if (allocated(atmo_lat)) deallocate(atmo_lat)
    if (allocated(atmo_wind10u)) deallocate(atmo_wind10u)
    if (allocated(atmo_wind10v)) deallocate(atmo_wind10v)
    if (allocated(atmo_obs_height)) deallocate(atmo_obs_height)
    if (allocated(atmo_radar_prop)) deallocate(atmo_radar_prop)

  end subroutine deallocate_atmosphere_vars
!##################################################################################################################################
  subroutine read_new_fill_variables(errorstatus)

    use settings, only: verbose, input_pathfile, radar_mode, active, read_turbulence_ascii
    use descriptor_file, only: moment_in_arr, n_hydro

    implicit none

    integer(kind=long), intent(out) :: errorstatus
    integer(kind=long) :: err
    character(len=14) :: nameOfRoutine = 'read_new_fill_variables'

! work variables
    real(kind=dbl), allocatable, dimension(:) :: work_xwp
    real(kind=dbl), allocatable, dimension(:,:) :: work_xwc
    real(kind=sgl) :: lfrac
    integer(kind=long) :: i, j, n_tot_moment, i_hydro, index_hydro

    err = 0

    if (verbose >= 3) call report(info,'Start of ', nameOfRoutine)

    n_tot_moment = 0
    do i_hydro = 1,n_hydro
      if (moment_in_arr(i_hydro) > 0.) n_tot_moment = n_tot_moment + 1
      if (moment_in_arr(i_hydro) > 5.) n_tot_moment = n_tot_moment + 1
    end do
    allocate(work_xwp(n_tot_moment+1))


! OPEN input file
    open(UNIT=14, FILE=TRIM(input_pathfile),&
    STATUS='OLD', iostat=err)
    read(14,*)
      do i = 1, atmo_ngridx
        do j = 1, atmo_ngridy
          read(14,*) atmo_year(i,j), atmo_month(i,j), atmo_day(i,j), atmo_time(i,j),&
                     atmo_nlyrs(i,j), atmo_model_i(i,j), atmo_model_j(i,j)
          if (atmo_input_type == 'lev') atmo_nlyrs(i,j) = atmo_nlyrs(i,j) - 1
          read(14,*) atmo_obs_height(i,j,:)         ! output heights [m]
          read(14,*) atmo_lat(i,j),       &         ! latitude [degree]
                     atmo_lon(i,j),       &         ! longitude [degree]
                     lfrac,               &         ! land sea fraction 1=land 0=ocean
                     atmo_wind10u(i,j),   &         ! 10 meter wind u comp. [m/s]
                     atmo_wind10v(i,j),   &         ! 10 meter wind v comp. [m/s]
                     atmo_groundtemp(i,j),&         ! ground temp. used for ground emission only [K]
                     atmo_hgt_lev(i,j,1)            ! ground height [m]
          sfc_type(i,j) = NINT(lfrac, long)

! READ column integrated water vapor and hydrometeors properties
          read(14,*) work_xwp
          atmo_iwv(i,j) = work_xwp(1)
          if (.not. read_turbulence_ascii) &
              allocate(work_xwc(n_tot_moment+4,atmo_nlyrs(i,j)))
! For radar moments or spectrum mode the atmospheric turbulence should be provided in the last column of the input file
          if (read_turbulence_ascii) &
              allocate(work_xwc(n_tot_moment+5,atmo_nlyrs(i,j)))
! READ lowest level variable (ONLY for "lev" input type)
          if (atmo_input_type == 'lev') read(14,*) atmo_hgt_lev(i,j,1),&
                     atmo_press_lev(i,j,1),atmo_temp_lev(i,j,1),atmo_relhum_lev(i,j,1)

! READ atmo_nlyrs(i,j) rows
          read(14,*) work_xwc

! FILL layer variables
          if (atmo_input_type == 'lay') then
            atmo_hgt(i,j,1:atmo_nlyrs(i,j))    = work_xwc(1,:)       ! layer height [m]
            atmo_press(i,j,1:atmo_nlyrs(i,j))  = work_xwc(2,:)       ! layer pressure [Pa]
            atmo_temp(i,j,1:atmo_nlyrs(i,j))   = work_xwc(3,:)       ! layer temperature [K]
            atmo_relhum(i,j,1:atmo_nlyrs(i,j)) = work_xwc(4,:)       ! layer relative humidity [%]
          endif
! FILL level variables
          if (atmo_input_type == 'lev') then
            atmo_hgt_lev(i,j,2:atmo_nlyrs(i,j)+1)    = work_xwc(1,:) ! layer height [m]
            atmo_press_lev(i,j,2:atmo_nlyrs(i,j)+1)  = work_xwc(2,:) ! layer pressure [Pa]
            atmo_temp_lev(i,j,2:atmo_nlyrs(i,j)+1)   = work_xwc(3,:) ! layer temperature [K]
            atmo_relhum_lev(i,j,2:atmo_nlyrs(i,j)+1) = work_xwc(4,:) ! layer relative humidity [%]
          endif


! FILL hydrometeor moments
          index_hydro = 5
          do i_hydro = 1,n_hydro
            if (moment_in_arr(i_hydro) == 1) then
              atmo_hydro_n(i,j,1:atmo_nlyrs(i,j),i_hydro)    = work_xwc(index_hydro,:)
              if (i_hydro == 1) atmo_hydro_n_column(i,j,i_hydro) = work_xwp(index_hydro)

            elseif (moment_in_arr(i_hydro) == 2) then
              atmo_hydro_reff(i,j,1:atmo_nlyrs(i,j),i_hydro) = work_xwc(index_hydro,:)
              if (i_hydro == 1) atmo_hydro_reff_column(i,j,i_hydro) = work_xwp(index_hydro)


            elseif (moment_in_arr(i_hydro) == 3) then
              atmo_hydro_q(i,j,1:atmo_nlyrs(i,j),i_hydro)    = work_xwc(index_hydro,:)
              if (i_hydro == 1) atmo_hydro_q_column(i,j,i_hydro) = work_xwp(index_hydro)


            elseif  (moment_in_arr(i_hydro) == 12) then
              atmo_hydro_n(i,j,1:atmo_nlyrs(i,j),i_hydro)    = work_xwc(index_hydro,:)
              atmo_hydro_reff(i,j,1:atmo_nlyrs(i,j),i_hydro) = work_xwc(index_hydro+1,:)
              if (i_hydro == 1) atmo_hydro_n_column(i,j,i_hydro) = work_xwp(index_hydro)
              if (i_hydro == 1) atmo_hydro_reff_column(i,j,i_hydro) = work_xwp(index_hydro+1)

            elseif (moment_in_arr(i_hydro) == 13) then
              atmo_hydro_n(i,j,1:atmo_nlyrs(i,j),i_hydro)    = work_xwc(index_hydro,:)
              atmo_hydro_q(i,j,1:atmo_nlyrs(i,j),i_hydro)    = work_xwc(index_hydro+1,:)
              if (i_hydro == 1) atmo_hydro_n_column(i,j,i_hydro) = work_xwp(index_hydro)
              if (i_hydro == 1) atmo_hydro_q_column(i,j,i_hydro) = work_xwp(index_hydro+1)


            elseif (moment_in_arr(i_hydro) == 23) then
              atmo_hydro_reff(i,j,1:atmo_nlyrs(i,j),i_hydro) = work_xwc(index_hydro,:)
              atmo_hydro_q(i,j,1:atmo_nlyrs(i,j),i_hydro)    = work_xwc(index_hydro+1,:)
              if (i_hydro == 1) atmo_hydro_reff_column(i,j,i_hydro) = work_xwp(index_hydro)
              if (i_hydro == 1) atmo_hydro_q_column(i,j,i_hydro) = work_xwp(index_hydro+1)
            endif


            if (moment_in_arr(i_hydro) < 5) index_hydro = index_hydro + 1
            if (moment_in_arr(i_hydro) > 5) index_hydro = index_hydro + 2
          enddo
! FILL turbulence for radar moments or spectrum mode
          if (read_turbulence_ascii) &
             atmo_airturb(i,j,1:atmo_nlyrs(i,j)) = work_xwc(index_hydro,:)

          deallocate(work_xwc)

        enddo
      enddo

    errorstatus = err
    if (verbose >= 3) call report(info,'End of ', nameOfRoutine)
    return

  end subroutine read_new_fill_variables
!##################################################################################################################################
  subroutine add_obs_height(errorstatus)

    use constants, only: r_v, r_d

    implicit none

    real(kind=dbl) :: e_sat_gg_water
    integer(kind=long), intent(out) :: errorstatus
    integer(kind=long) :: err
    character(len=14) :: nameOfRoutine = 'add_obs_height'

! work variables
    real(kind=dbl) :: derivative, dh, dh_1, var_newlev
    integer(kind=long)  :: i, j, k, nz, lev_1, lev_2, lev_use, lay_use, ii

    err = 0

    if (verbose >= 3) call report(info,'Start of ', nameOfRoutine)

    do i = 1, atmo_ngridx
      do j = 1, atmo_ngridy
        do k = 1, noutlevels
        if (atmo_obs_height(i,j,k) < atmo_hgt_lev(i,j,1)) then
	  atmo_obs_height(i,j,k) = atmo_hgt_lev(i,j,1)
	  call report(info, 'Setting observation height to lowest atmo level', nameOfRoutine)
	end if
! FIND if a new level is needed: obs_height > heighest level .and. abs(hgt_lev - obs_height) > 1 m
        if (atmo_hgt_lev(i,j,atmo_nlyrs(i,j)) > atmo_obs_height(i,j,k) .and. &
            minval(abs(atmo_hgt_lev(i,j,:) - atmo_obs_height(i,j,k))) > 1.) then
! FIND first level above obs_height
          do nz = 1, atmo_nlyrs(i,j)
            if (atmo_hgt_lev(i,j,nz) > atmo_obs_height(i,j,k)) then
              lev_1 = nz - 1
              lev_2 = nz
              lay_use = lev_1
              if (atmo_hgt(i,j,lay_use) < atmo_obs_height(i,j,k)) lev_use = lev_2
              if (atmo_hgt(i,j,lay_use) > atmo_obs_height(i,j,k)) lev_use = lev_1
              exit
            endif
          enddo
! CALCULATE the variable on the NEW LEVEL
          dh_1 = atmo_hgt_lev(i,j,lev_use) - atmo_hgt(i,j,lay_use) ! used for the derivative
          dh = atmo_obs_height(i,j,k) - atmo_hgt(i,j,lay_use)        ! height increment
          ! INSERT obs_height as the new level
          atmo_hgt_lev(i,j,atmo_nlyrs(i,j)+2:lev_2+1:-1) = atmo_hgt_lev(i,j,atmo_nlyrs(i,j)+1:lev_2:-1)
          atmo_hgt_lev(i,j,lev_2) = atmo_obs_height(i,j,k)

          derivative = (atmo_temp_lev(i,j,lev_use) - atmo_temp(i,j,lay_use)) / dh_1
          var_newlev = atmo_temp(i,j,lay_use) + derivative * dh
          atmo_temp_lev(i,j,atmo_nlyrs(i,j)+2:lev_2+1:-1) = atmo_temp_lev(i,j,atmo_nlyrs(i,j)+1:lev_2:-1)
          atmo_temp_lev(i,j,lev_2) = var_newlev

          derivative = (atmo_relhum_lev(i,j,lev_use) - atmo_relhum(i,j,lay_use)) / dh_1
          var_newlev = atmo_relhum(i,j,lay_use) + derivative * dh
          atmo_relhum_lev(i,j,atmo_nlyrs(i,j)+2:lev_2+1:-1) = atmo_relhum_lev(i,j,atmo_nlyrs(i,j)+1:lev_2:-1)
          atmo_relhum_lev(i,j,lev_2) = var_newlev

          derivative = (log(atmo_press_lev(i,j,lev_use)) - log(atmo_press(i,j,lay_use))) / dh_1
          var_newlev = exp ( log(atmo_press(i,j,lay_use)) + derivative * dh )
          atmo_press_lev(i,j,atmo_nlyrs(i,j)+2:lev_2+1:-1) = atmo_press_lev(i,j,atmo_nlyrs(i,j)+1:lev_2:-1)
          atmo_press_lev(i,j,lev_2) = var_newlev

! recalculate delta_hgt_lev
          atmo_delta_hgt_lev(i,j,atmo_nlyrs(i,j)+1:lev_2+1:-1)  = atmo_delta_hgt_lev(i,j,atmo_nlyrs(i,j):lev_2:-1)
          atmo_delta_hgt_lev(i,j,lev_1)  = atmo_hgt_lev(i,j,lev_1+1) - atmo_hgt_lev(i,j,lev_1)
          atmo_delta_hgt_lev(i,j,lev_2)  = atmo_hgt_lev(i,j,lev_2+1) - atmo_hgt_lev(i,j,lev_2)

! now the LAYER variables
          atmo_relhum(i,j,atmo_nlyrs(i,j)+1:lay_use+2:-1)         = atmo_relhum(i,j,atmo_nlyrs(i,j):lay_use+1:-1)
          atmo_press(i,j,atmo_nlyrs(i,j)+1:lay_use+2:-1)          = atmo_press(i,j,atmo_nlyrs(i,j):lay_use+1:-1)
          atmo_temp(i,j,atmo_nlyrs(i,j)+1:lay_use+2:-1)           = atmo_temp(i,j,atmo_nlyrs(i,j):lay_use+1:-1)
          atmo_hgt(i,j,atmo_nlyrs(i,j)+1:lay_use+2:-1)            = atmo_hgt(i,j,atmo_nlyrs(i,j):lay_use+1:-1)
          atmo_airturb(i,j,atmo_nlyrs(i,j)+1:lay_use+2:-1)        = atmo_airturb(i,j,atmo_nlyrs(i,j):lay_use+1:-1)
          atmo_airturb(i,j,lay_use+1)                             = atmo_airturb(i,j,lay_use) ! Keep the same value for the 2 children layer
          atmo_turb_edr(i,j,atmo_nlyrs(i,j)+1:lay_use+2:-1)       = atmo_turb_edr(i,j,atmo_nlyrs(i,j):lay_use+1:-1)
          atmo_turb_edr(i,j,lay_use+1)                            = atmo_turb_edr(i,j,lay_use) ! Keep the same value for the 2 children layer
          atmo_wind_uv(i,j,atmo_nlyrs(i,j)+1:lay_use+2:-1)        = atmo_wind_uv(i,j,atmo_nlyrs(i,j):lay_use+1:-1)
          atmo_wind_uv(i,j,lay_use+1)                             = atmo_wind_uv(i,j,lay_use) ! Keep the same value for the 2 children layer
          atmo_wind_w(i,j,atmo_nlyrs(i,j)+1:lay_use+2:-1)         = atmo_wind_w(i,j,atmo_nlyrs(i,j):lay_use+1:-1)
          atmo_wind_w(i,j,lay_use+1)                              = atmo_wind_w(i,j,lay_use) ! Keep the same value for the 2 children layer
          atmo_vapor_pressure(i,j,atmo_nlyrs(i,j)+1:lay_use+2:-1) = atmo_vapor_pressure(i,j,atmo_nlyrs(i,j):lay_use+1:-1)
          atmo_rho_vap(i,j,atmo_nlyrs(i,j)+1:lay_use+2:-1)        = atmo_rho_vap(i,j,atmo_nlyrs(i,j):lay_use+1:-1)
          atmo_q_hum(i,j,atmo_nlyrs(i,j)+1:lay_use+2:-1)          = atmo_q_hum(i,j,atmo_nlyrs(i,j):lay_use+1:-1)
! Hydrometeors: for the 2 children layers, use the same mass [kg/kg] and number [#/kg] concentration as the father ones
!               this introduce a small inconsistency since the air density can change due to T and q changes
!               !!!!! CHECK that it is a VERY small error
          atmo_hydro_q(i,j,atmo_nlyrs(i,j)+1:lay_use+2:-1,:)      = atmo_hydro_q(i,j,atmo_nlyrs(i,j):lay_use+1:-1,:)
          atmo_hydro_q(i,j,lay_use+1,:)                           = atmo_hydro_q(i,j,lay_use,:)
          atmo_hydro_reff(i,j,atmo_nlyrs(i,j)+1:lay_use+2:-1,:)   = atmo_hydro_reff(i,j,atmo_nlyrs(i,j):lay_use+1:-1,:)
          atmo_hydro_reff(i,j,lay_use+1,:)                        = atmo_hydro_reff(i,j,lay_use,:)
          atmo_hydro_n(i,j,atmo_nlyrs(i,j)+1:lay_use+2:-1,:)      = atmo_hydro_n(i,j,atmo_nlyrs(i,j):lay_use+1:-1,:)
          atmo_hydro_n(i,j,lay_use+1,:)                           = atmo_hydro_n(i,j,lay_use,:)

          do ii = 0, 1
            atmo_relhum(i,j,lay_use+ii) = 0.5_dbl * (atmo_relhum_lev(i,j,lev_1+ii) + atmo_relhum_lev(i,j,lev_1+1+ii))
            atmo_press(i,j,lay_use+ii)  = &
            exp( 0.5_dbl * (log(atmo_press_lev(i,j,lev_1+ii)) + log(atmo_press_lev(i,j,lev_1+1+ii))))
            atmo_temp(i,j,lay_use+ii)   = 0.5_dbl * (atmo_temp_lev(i,j,lev_1+ii) + atmo_temp_lev(i,j,lev_1+1+ii))
            atmo_hgt(i,j,lay_use+ii)    = 0.5_dbl * (atmo_hgt_lev(i,j,lev_1+ii) + atmo_hgt_lev(i,j,lev_1+1+ii))
            atmo_vapor_pressure(i,j,lay_use+ii) = atmo_relhum(i,j,lay_use+ii) * e_sat_gg_water(atmo_temp(i,j,lay_use+ii)) ! Pa
            atmo_rho_vap(i,j,lay_use+ii)        = atmo_vapor_pressure(i,j,lay_use+ii)/(atmo_temp(i,j,lay_use+ii) * r_v)  ! [kg/m3]
            atmo_q_hum(i,j,lay_use+ii)          = r_v/r_d*atmo_vapor_pressure(i,j,lay_use+ii)/&
                (atmo_press(i,j,lay_use+ii) - (1._dbl- r_v/r_d) * atmo_vapor_pressure(i,j,lay_use+ii))  ! [kg/kg]
          enddo
          atmo_nlyrs(i,j) = atmo_nlyrs(i,j)+1

        endif
        end do
      enddo
    enddo

    errorstatus = err
    if (verbose >= 3) call report(info,'End of ', nameOfRoutine)
    return

  end subroutine add_obs_height
!##################################################################################################################################
  subroutine read_classic_fill_variables(errorstatus)

    use settings, only: verbose, input_pathfile

    implicit none

    integer(kind=long), intent(out) :: errorstatus
    integer(kind=long) :: err
    character(len=14) :: nameOfRoutine = 'read_classic_fill_variables'

! work variables
    real(kind=dbl), allocatable, dimension(:,:) :: work_xwc
    real(kind=dbl) :: dum
    integer(kind=long)  :: i, j
    real(kind=sgl) :: lfrac

    err = 0

    if (verbose >= 3) call report(info,'Start of ', nameOfRoutine)

! OPEN input file
    open(UNIT=14, FILE=TRIM(input_pathfile),&
    STATUS='OLD', iostat=err)
    read(14,*) atmo_year(1,1), atmo_month(1,1), atmo_day(1,1), atmo_time(1,1),dum,dum,atmo_nlyrs(1,1)
    atmo_year(:,:)  = atmo_year(1,1)
    atmo_month(:,:) = atmo_month(1,1)
    atmo_day(:,:)   = atmo_day(1,1)
    atmo_time(:,:)  = atmo_time(1,1)
    atmo_nlyrs(:,:) = atmo_nlyrs(1,1)
    atmo_obs_height(:,:,1) = 833000.
    atmo_obs_height(:,:,2) = 0.
    if (noutlevels > 2) call report(info, 'Only 2 outputlevels for classic profiles', nameOfRoutine)


      do i = 1, atmo_ngridx
        do j = 1, atmo_ngridy
          read(14,*) atmo_model_i(i,j), atmo_model_j(i,j)
          read(14,*) atmo_lat(i,j),       &         ! latitude [degree]
                     atmo_lon(i,j),       &         ! longitude [degree]
                     lfrac,               &         ! land sea fraction 1=land 0=ocean
                     atmo_wind10u(i,j),   &         ! 10 meter wind u comp. [m/s]
                     atmo_wind10v(i,j)              ! 10 meter wind v comp. [m/s]
          sfc_type(i,j) = NINT(lfrac, long)

! READ column integrated water vapor and hydrometeors properties
          read(14,*) atmo_iwv(i,j), atmo_hydro_q_column(i,j,:)

! READ lowest level variables
          read(14,*) atmo_hgt_lev(i,j,1), atmo_press_lev(i,j,1), atmo_temp_lev(i,j,1), atmo_relhum_lev(i,j,1)
          atmo_groundtemp(i,j) = atmo_temp_lev(i,j,1)

! Work variable where the layer/level profile variables will be read-in
          allocate(work_xwc(9,atmo_nlyrs(i,j)))

! READ atmo_nlyrs(i,j) rows
          read(14,*) work_xwc

! FILL level variables
          atmo_hgt_lev(i,j,2:atmo_nlyrs(i,j)+1)    = work_xwc(1,:) ! layer height [m]
          atmo_press_lev(i,j,2:atmo_nlyrs(i,j)+1)  = work_xwc(2,:) ! layer pressure [Pa]
          atmo_temp_lev(i,j,2:atmo_nlyrs(i,j)+1)   = work_xwc(3,:) ! layer temperature [K]
          atmo_relhum_lev(i,j,2:atmo_nlyrs(i,j)+1) = work_xwc(4,:) ! layer relative humidity [%]

! FILL hydrometeor moments
          atmo_hydro_q(i,j,1:atmo_nlyrs(i,j),:)    = transpose(work_xwc(5:9,:))
          deallocate(work_xwc)

        enddo
      enddo

    errorstatus = err
    if (verbose >= 3) call report(info,'End of ', nameOfRoutine)
    return

  end subroutine read_classic_fill_variables
!##################################################################################################################################
  subroutine fillMissing_atmosphere_vars(errorstatus)

    use descriptor_file, only: n_hydro, moment_in_arr
    use constants, only: r_v, r_d

    implicit none

    integer :: nz, nx, ny
    real(kind=dbl) :: e_sat_gg_water
    integer,dimension(9) :: timestamp

    integer(kind=long), intent(out) :: errorstatus
    integer(kind=long) :: err
    character(len=80) :: msg
    character(len=30) :: nameOfRoutine = 'fillMissing_atmosphere_vars'

    integer(kind=long) :: i_hydro

    err = 0

    if (verbose >= 3) call report(info,'Start of ', nameOfRoutine)
    err = 0
    call assert_true(err,(atmo_ngridx>0),&
        "atmo_ngridx must be greater zero")
    call assert_true(err,(atmo_ngridy>0),&
        "atmo_ngridy must be greater zero")
    call assert_true(err,(all(atmo_nlyrs>0)),&
        "atmo_nlyrs must be greater zero")
    call assert_true(err,(noutlevels>0),&
        "noutlevels must be greater zero")
    if (err > 0) then
        errorstatus = fatal
        msg = "assertation error"
        call report(errorstatus, msg, nameOfRoutine)
        return
    end if

    do nx = 1, atmo_ngridx
      do ny = 1, atmo_ngridy
          if (verbose >= 5) print*,  "processing", nx, ny
          !these one depend not on z:
          if (atmo_unixtime(nx,ny) /= -9999) then
                call GMTIME(atmo_unixtime(nx,ny),timestamp)
                write(atmo_year(nx,ny),"(i4.4)") timestamp(6)+1900
                write(atmo_month(nx,ny),"(i2.2)") timestamp(5)+1
                write(atmo_day(nx,ny),"(i2.2)") timestamp(4)
                write(atmo_time(nx,ny)(1:2),"(i2.2)") timestamp(3)
                write(atmo_time(nx,ny)(3:4),"(i2.2)") timestamp(2)
          end if



          call assert_true(err,(all(atmo_hgt_lev(nx,ny,1:atmo_nlyrs(nx,ny)+1)>-500) &
              .or. all(atmo_hgt(nx,ny,1:atmo_nlyrs(nx,ny))>-500)),&
              "hgt or hgt_lev must be greater -500 (Depression in northern Egypt within the CORDEX domain :-))")
          if (err > 0) then
              errorstatus = fatal
              msg = "assertation error"
              call report(errorstatus, msg, nameOfRoutine)
              return
          end if
          ! first make sure that hgt is present
          do nz = 1, atmo_nlyrs(nx,ny)
            if (isnan(atmo_hgt(nx,ny,nz))) &
                  atmo_hgt(nx,ny,nz) = 0.5_dbl * (atmo_hgt_lev(nx,ny,nz) + atmo_hgt_lev(nx,ny,nz+1))
          end do


          ! make the level variables which are needed
          ! not needed anywhere as of today are press_lev and relhum_lev:

          if (isnan(atmo_hgt_lev(nx,ny,1))) &
            atmo_hgt_lev(nx,ny,1) = atmo_hgt(nx,ny,1) - &
            (atmo_hgt(nx,ny,2) - atmo_hgt(nx,ny,1))*0.5_dbl
          if (isnan(atmo_temp_lev(nx,ny,1))) &
            atmo_temp_lev(nx,ny,1) = atmo_temp(nx,ny,1) - &
            (atmo_temp(nx,ny,2) - atmo_temp(nx,ny,1)) / (atmo_hgt(nx,ny,2) - atmo_hgt(nx,ny,1)) * &
            (atmo_hgt(nx,ny,1) - atmo_hgt_lev(nx,ny,1))
          if (isnan(atmo_relhum_lev(nx,ny,1))) &
            atmo_relhum_lev(nx,ny,1) = atmo_relhum(nx,ny,1) - &
            (atmo_relhum(nx,ny,2) - atmo_relhum(nx,ny,1)) / (atmo_hgt(nx,ny,2) - atmo_hgt(nx,ny,1)) * &
            (atmo_hgt(nx,ny,1) - atmo_hgt_lev(nx,ny,1))
          if (isnan(atmo_press_lev(nx,ny,1))) &
            atmo_press_lev(nx,ny,1) = exp ( log(atmo_press(nx,ny,1)) - &
            (log(atmo_press(nx,ny,2)) - log(atmo_press(nx,ny,1))) / (atmo_hgt(nx,ny,2) - atmo_hgt(nx,ny,1)) * &
            (atmo_hgt(nx,ny,1) - atmo_hgt_lev(nx,ny,1)))

          do nz = 2, atmo_nlyrs(nx,ny)
           if (isnan(atmo_temp_lev(nx,ny,nz))) &
              atmo_temp_lev(nx,ny,nz) = 0.5_dbl * (atmo_temp(nx,ny,nz) + atmo_temp(nx,ny,nz-1))
            if (isnan(atmo_hgt_lev(nx,ny,nz))) &
              atmo_hgt_lev(nx,ny,nz) = 0.5_dbl * (atmo_hgt(nx,ny,nz) + atmo_hgt(nx,ny,nz-1))
            if (isnan(atmo_relhum_lev(nx,ny,nz))) &
              atmo_relhum_lev(nx,ny,nz) = 0.5_dbl * (atmo_relhum(nx,ny,nz) + atmo_relhum(nx,ny,nz-1))
            if (isnan(atmo_press_lev(nx,ny,nz))) &
              atmo_press_lev(nx,ny,nz) = exp (0.5_dbl * (log(atmo_press(nx,ny,nz)) + log(atmo_press(nx,ny,nz-1))))
          end do
          nz = atmo_nlyrs(nx,ny) +1
          if (isnan(atmo_hgt_lev(nx,ny,nz))) &
            atmo_hgt_lev(nx,ny,nz) = atmo_hgt(nx,ny,nz-1) + &
            (atmo_hgt(nx,ny,nz-1) - atmo_hgt(nx,ny,nz-2))*0.5_dbl
          if (isnan(atmo_temp_lev(nx,ny,nz))) &
            atmo_temp_lev(nx,ny,nz) = atmo_temp(nx,ny,nz-1) + &
            (atmo_temp(nx,ny,nz-1) - atmo_temp(nx,ny,nz-2)) / (atmo_hgt(nx,ny,nz-1) - atmo_hgt(nx,ny,nz-2)) * &
            (atmo_hgt_lev(nx,ny,nz) - atmo_hgt(nx,ny,nz-1))
          if (isnan(atmo_relhum_lev(nx,ny,nz))) &
            atmo_relhum_lev(nx,ny,nz) = atmo_relhum(nx,ny,nz-1) + &
            (atmo_relhum(nx,ny,nz-1) - atmo_relhum(nx,ny,nz-2)) / (atmo_hgt(nx,ny,nz-1) - atmo_hgt(nx,ny,nz-2)) * &
            (atmo_hgt_lev(nx,ny,nz) - atmo_hgt(nx,ny,nz-1))
          if (isnan(atmo_press_lev(nx,ny,nz))) &
            atmo_press_lev(nx,ny,nz) = exp ( log(atmo_press(nx,ny,nz-1)) + &
            (log(atmo_press(nx,ny,nz-1)) - log(atmo_press(nx,ny,nz-2))) / (atmo_hgt(nx,ny,nz-1) - atmo_hgt(nx,ny,nz-2)) * &
            (atmo_hgt_lev(nx,ny,nz) - atmo_hgt(nx,ny,nz-1)) )

          !now the layers variables
          do nz = 1, atmo_nlyrs(nx,ny)

          ! first test the layers variables for nan and calculate them otherwise
          if (isnan(atmo_temp(nx,ny,nz))) &
                atmo_temp(nx,ny,nz) = 0.5_dbl * (atmo_temp_lev(nx,ny,nz) + atmo_temp_lev(nx,ny,nz+1))
          if (isnan(atmo_relhum(nx,ny,nz))) &
                atmo_relhum(nx,ny,nz) = 0.5_dbl * (atmo_relhum_lev(nx,ny,nz) + atmo_relhum_lev(nx,ny,nz+1))
          if (isnan(atmo_press(nx,ny,nz))) &
                atmo_press(nx,ny,nz) = exp(0.5_dbl * (log(atmo_press_lev(nx,ny,nz)) + log(atmo_press_lev(nx,ny,nz+1))))  ! [Pa]
          if (isnan(atmo_hgt(nx,ny,nz))) &
                atmo_hgt(nx,ny,nz) = (atmo_hgt_lev(nx,ny,nz)+atmo_hgt_lev(nx,ny,nz+1))*0.5_dbl
          if (isnan(atmo_delta_hgt_lev(nx,ny,nz))) &
                atmo_delta_hgt_lev(nx,ny,nz) = atmo_hgt_lev(nx,ny,nz+1) - atmo_hgt_lev(nx,ny,nz)

         ! now the ones which depend on several
          if (isnan(atmo_vapor_pressure(nx,ny,nz))) &
                atmo_vapor_pressure(nx,ny,nz) = atmo_relhum(nx,ny,nz) * &
                e_sat_gg_water(atmo_temp(nx,ny,nz)) ! Pa
          if (isnan(atmo_rho_vap(nx,ny,nz))) &
                atmo_rho_vap(nx,ny,nz) = atmo_vapor_pressure(nx,ny,nz)/(atmo_temp(nx,ny,nz) * r_v)  ! [kg/m3]
          if (isnan(atmo_q_hum(nx,ny,nz))) &
                atmo_q_hum(nx,ny,nz) = r_v/r_d*atmo_vapor_pressure(nx,ny,nz)/&
                (atmo_press(nx,ny,nz) - (1._dbl- r_v/r_d) * atmo_vapor_pressure(nx,ny,nz))  ! [kg/kg]


        end do
    !for ground temp, simply use the lowes avalable one if not provided
    if (isnan(atmo_groundtemp(nx,ny))) &
        atmo_groundtemp(nx,ny) = atmo_temp_lev(nx,ny,1)

    !for atmo_obs_height, simply use the nml value
    if (isnan(atmo_obs_height(nx,ny,noutlevels))) &
        atmo_obs_height(nx,ny,:) = (/833000.,0./)

    !test whether we still have nans in our data!
! 3D variable
    err = 0
    call assert_true(err,atmo_nlyrs(nx,ny) <= atmo_max_nlyrs ,&
        "atmo_nlyrs(nx,ny) larger than atmo_max_nlyrs")
    call assert_false(err,ANY(ISNAN(atmo_temp_lev(nx,ny,1:atmo_nlyrs(nx,ny)+1))),&
        "found nan in atmo_temp_lev")
    call assert_false(err,ANY(ISNAN(atmo_hgt_lev(nx,ny,1:atmo_nlyrs(nx,ny)+1))),&
        "found nan in atmo_hgt_lev")
    call assert_true(err,atmo_hgt_lev(nx,ny,atmo_nlyrs(nx,ny)+1)>atmo_hgt_lev(nx,ny,1),&
        "atmo_hgt_lev must be defined defined bottom-up")
    call assert_false(err,ANY(ISNAN(atmo_hgt(nx,ny,1:atmo_nlyrs(nx,ny)))),&
        "found nan in atmo_hgt")
    call assert_false(err,ANY(ISNAN(atmo_relhum(nx,ny,1:atmo_nlyrs(nx,ny)))),&
        "found nan in atmo_relhum")
    call assert_false(err,ANY(ISNAN(atmo_press(nx,ny,1:atmo_nlyrs(nx,ny)))),&
        "found nan in atmo_press")
    call assert_false(err,ANY(ISNAN(atmo_temp(nx,ny,1:atmo_nlyrs(nx,ny)))),&
        "found nan in atmo_temp")
    call assert_false(err,ANY(ISNAN(atmo_vapor_pressure(nx,ny,1:atmo_nlyrs(nx,ny)))),&
        "found nan in atmo_vapor_pressure")
    call assert_false(err,ANY(ISNAN(atmo_rho_vap(nx,ny,1:atmo_nlyrs(nx,ny)))),&
        "found nan in atmo_rho_vap")
    call assert_false(err,ANY(ISNAN(atmo_q_hum(nx,ny,1:atmo_nlyrs(nx,ny)))),&
        "found nan in atmo_q_hum")
    call assert_false(err,ANY(ISNAN(atmo_hgt(nx,ny,1:atmo_nlyrs(nx,ny)))),&
        "found nan in atmo_hgt")
    call assert_false(err,ANY(ISNAN(atmo_delta_hgt_lev(nx,ny,1:atmo_nlyrs(nx,ny)))),&
        "found nan in atmo_delta_hgt_lev")

!     call assert_false(err,ANY(ISNAN(atmo_wind_w(nx,ny,1:atmo_nlyrs(nx,ny)))),&
!         "found nan in atmo_wind_w")
    call assert_false(err,ANY(ISNAN(atmo_relhum_lev(nx,ny,1:atmo_nlyrs(nx,ny)+1))),&
        "found nan in atmo_relhum_lev")
    call assert_false(err,ANY(ISNAN(atmo_press_lev(nx,ny,1:atmo_nlyrs(nx,ny)+1))),&
        "found nan in atmo_press_lev")
    call assert_true(err,ALL(moment_in_arr >= 0),&
        "found negative value in moment_in_arr")

    do i_hydro = 1,n_hydro
      if ((moment_in_arr(i_hydro) == 1) .or. (moment_in_arr(i_hydro) == 12) .or. (moment_in_arr(i_hydro) == 13)) &
       call assert_false(err,ANY(ISNAN(atmo_hydro_n(nx,ny,1:atmo_nlyrs(nx,ny),i_hydro))),&
          "found nan in atmo_hydro_n")
      if ((moment_in_arr(i_hydro) == 2) .or. (moment_in_arr(i_hydro) == 12 ).or. (moment_in_arr(i_hydro) == 23)) &
       call assert_false(err,ANY(ISNAN(atmo_hydro_reff(nx,ny,1:atmo_nlyrs(nx,ny),i_hydro))),&
          "found nan in atmo_hydro_reff")
      if ((moment_in_arr(i_hydro) == 3) .or. (moment_in_arr(i_hydro) == 13) .or. (moment_in_arr(i_hydro) == 23)) &
       call assert_false(err,ANY(ISNAN(atmo_hydro_q(nx,ny,1:atmo_nlyrs(nx,ny),i_hydro))),&
          "found nan in atmo_hydro_q")
    enddo
      end do
    end do
! 2D variables
    call assert_false(err,ANY(ISNAN(atmo_lon(:,:))),&
        "found nan in atmo_lon")
    call assert_false(err,ANY(ISNAN(atmo_lat(:,:))),&
        "found nan in atmo_lat")
    call assert_false(err,ANY(ISNAN(atmo_wind10u(:,:))),&
        "found nan in atmo_wind10u")
    call assert_false(err,ANY(ISNAN(atmo_wind10v(:,:))),&
        "found nan in atmo_wind10v")
    call assert_false(err,ANY(ISNAN(atmo_obs_height(:,:,:))),&
        "found nan in atmo_obs_height")
    call assert_false(err,ANY(ISNAN(atmo_groundtemp(:,:))),&
        "found nan in atmo_groundtemp")
!     call assert_false(err,ANY(ISNAN(atmo_iwv(:,:))),&
!         "found nan in atmo_iwv")

    if (err > 0) then
        errorstatus = fatal
        msg = "assertation error"
        call report(errorstatus, msg, nameOfRoutine)
        return
    end if

   if (verbose >= 5) call print_vars_atmosphere()

    errorstatus = err
    if (verbose >= 3) call report(info,'End of ', nameOfRoutine)
    return

  end subroutine fillMissing_atmosphere_vars
!##################################################################################################################################
  subroutine print_out_layer(i,j)
    !for debuging prurposes

    implicit none

    integer(kind=long) :: i, j, nz

    write(6,'(7a12)') 'year', 'month', 'day', 'time', 'nlyrs', 'model_i', 'model_j'
    write(6,'(4a12,3i12)') atmo_year(i,j), atmo_month(i,j), atmo_day(i,j), atmo_time(i,j), atmo_nlyrs(i,j),&
                           atmo_model_i(i,j), atmo_model_j(i,j)
    write(6,*) 'atmo_obs_height'
    write(6,*) atmo_obs_height(i,j,:)
    write(6,'(8a12)') 'lat', 'lon','lfrac','wind10u','wind10v','groundtemp','ground_hgt'
    write(6,'(7f12.4)') atmo_lat(i,j), atmo_lon(i,j),REAL(sfc_type(i,j)),atmo_wind10u(i,j),atmo_wind10v(i,j),&
                        atmo_groundtemp(i,j),atmo_hgt_lev(i,j,1)
    write(6,'(6a12)') 'lay_number','height','pressure','temp.','RH','air_turb'
    do nz=1,atmo_nlyrs(i,j)
      write(6,'(i12,5f12.4)') nz,atmo_hgt(i,j,nz), atmo_press(i,j,nz), &
                              atmo_temp(i,j,nz), atmo_relhum(i,j,nz), atmo_airturb(i,j,nz), atmo_wind_w(i,j,nz)
    enddo
    write(6,'(8a12)') 'lay_number','height','Q_hydro1','Q_hydro2','Q_hydro3','Q_hydro4','Q_hydro5','Q_hydro6'
    do nz=1,atmo_nlyrs(i,j)
      write(6,'(i12,7f12.4)') nz,atmo_hgt(i,j,nz),atmo_hydro_q(i,j,nz,:)
    enddo
    write(6,'(8a12)') 'lay_number','height','N_hydro1','N_hydro2','N_hydro3','N_hydro4','N_hydro5','N_hydro6'
    do nz=1,atmo_nlyrs(i,j)
      write(6,'(i12,7f12.4)') nz,atmo_hgt(i,j,nz),atmo_hydro_n(i,j,nz,:)
    enddo
    write(6,'(8a12)') 'lay_number','height','Ref_hydro1','Ref_hydro2','Ref_hydro3','Ref_hydro4','Ref_hydro5','Ref_hydro6'
    do nz=1,atmo_nlyrs(i,j)
      write(6,'(i12,7f12.4)') nz,atmo_hgt(i,j,nz),atmo_hydro_reff(i,j,nz,:)
    enddo

    return

  end subroutine print_out_layer
!##################################################################################################################################
  subroutine print_out_level(i,j)
    !for debuging prurposes

    implicit none

    integer(kind=long) :: i, j, nz

    write(6,'(7a12)') 'year', 'month', 'day', 'time', 'nlev', 'model_i', 'model_j'
    write(6,'(4a12,3i12)') atmo_year(i,j), atmo_month(i,j), atmo_day(i,j), atmo_time(i,j), atmo_nlyrs(i,j)+1,&
                           atmo_model_i(i,j), atmo_model_j(i,j)
    write(6,*) 'atmo_obs_height'
    write(6,*) atmo_obs_height(i,j,:)
    write(6,'(8a12)') 'lat', 'lon','lfrac','wind10u','wind10v','groundtemp','ground_hgt'
    write(6,'(8f12.4)') atmo_lat(i,j), atmo_lon(i,j),REAL(sfc_type(i,j)),atmo_wind10u(i,j),atmo_wind10v(i,j),&
                        atmo_groundtemp(i,j),atmo_hgt_lev(i,j,1)
    write(6,'(5a12)') 'lev_number','height','pressure','temp.','RH'
    do nz=1,atmo_nlyrs(i,j)+1
      write(6,'(i12,4f12.4)') nz,atmo_hgt_lev(i,j,nz), atmo_press_lev(i,j,nz), atmo_temp_lev(i,j,nz), atmo_relhum_lev(i,j,nz)
    enddo
!     write(6,'(7a12)') 'height','Q_hydro1','Q_hydro2','Q_hydro3','Q_hydro4','Q_hydro5','Q_hydro6'
!     do nz=1,atmo_nlyrs(i,j)
!       write(6,'(7f12.4)') atmo_hgt_lev(i,j,nz+1),atmo_hydro_q(i,j,nz,:)
!     enddo
!     write(6,'(7a12)') 'height','N_hydro1','N_hydro2','N_hydro3','N_hydro4','N_hydro5','N_hydro6'
!     do nz=1,atmo_nlyrs(i,j)
!       write(6,'(7f12.4)') atmo_hgt_lev(i,j,nz+1),atmo_hydro_n(i,j,nz,:)
!     enddo
!     write(6,'(7a12)') 'height','Ref_hydro1','Ref_hydro2','Ref_hydro3','Ref_hydro4','Ref_hydro5','Ref_hydro6'
!     do nz=1,atmo_nlyrs(i,j)
!       write(6,'(7f12.4)') atmo_hgt_lev(i,j,nz+1),atmo_hydro_reff(i,j,nz,:)
!     enddo


    return

  end subroutine print_out_level
!##################################################################################################################################
  subroutine print_vars_atmosphere()
    !for debuging prurposes


    print*, "atmo_nlyrs", atmo_nlyrs
    print*, "atmo_unixtime", atmo_unixtime

    print*, "atmo_relhum_lev", atmo_relhum_lev
    print*, "atmo_press_lev", atmo_press_lev
    print*, "atmo_temp_lev", atmo_temp_lev
    print*, "atmo_hgt_lev", atmo_hgt_lev

    print*, "atmo_relhum", atmo_relhum
    print*, "atmo_press", atmo_press
    print*, "atmo_vapor_pressure", atmo_vapor_pressure
    print*, "atmo_rho_vap", atmo_rho_vap
    print*, "atmo_q_hum", atmo_q_hum
    print*, "atmo_temp", atmo_temp
    print*, "atmo_hgt", atmo_hgt
    print*, "atmo_delta_hgt_lev", atmo_delta_hgt_lev
    print*, "atmo_airturb", atmo_airturb
    print*, "atmo_wind_w", atmo_wind_w
    print*, "atmo_wind_uv", atmo_wind_uv
    print*, "atmo_turb_edr", atmo_turb_edr

    print*, "atmo_hydro_q", atmo_hydro_q
    print*, "atmo_hydro_reff", atmo_hydro_reff
    print*, "atmo_hydro_n", atmo_hydro_n

    print*, "atmo_groundtemp", atmo_groundtemp
    print*, "atmo_iwv", atmo_iwv

    print*, "atmo_month", atmo_month
    print*, "atmo_day", atmo_day
    print*, "atmo_year", atmo_year
    print*, "atmo_time", atmo_time
!     print*, "atmo_deltax", atmo_deltax
!     print*, "atmo_deltay", atmo_deltay
    print*, "atmo_model_i", atmo_model_i
    print*, "atmo_model_j", atmo_model_j
    print*, "atmo_lon", atmo_lon
    print*, "atmo_lat", atmo_lat
    print*, "atmo_wind10u", atmo_wind10u
    print*, "atmo_wind10v", atmo_wind10v
    print*, "atmo_obs_height", atmo_obs_height
    print*, "atmo_radar_prop", atmo_radar_prop

    print*, "sfc_type", sfc_type
    print*, "sfc_model", sfc_model
    print*, "sfc_refl", sfc_refl
    print*, "sfc_salinity", sfc_salinity
    print*, "sfc_slf", sfc_slf
    print*, "sfc_sif", sfc_sif

  end subroutine print_vars_atmosphere

end module vars_atmosphere
