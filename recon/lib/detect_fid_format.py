#****************************************************************
def detect_fid_format(filename,nframe,nseg,nslice,N_pe,N_fe,datasize,time_interp,phase_correct,pulse_sequence):
#****************************************************************
    """
    Purpose:  Determine format of fid file.
    nframe: Number of frames specified in procpar file.
    nslice: Number of slices.
    N_pe: Number of phase encodes including navigator echoes.
    N_fe: Number of frequency codes. 
    datasize: Number of bytes per value.
    """
    MAIN_HDR= 32
    SUB_HDR= 28

#   Compute number of bytes per frame
    len_asems_ncsnn = N_pe*(SUB_HDR + 2*nslice*N_fe*datasize)
    len_asems_nccnn = N_pe*nslice*(SUB_HDR + 2*N_fe*datasize)
    len_compressed = SUB_HDR + 2*nslice*N_fe*N_pe*datasize
    len_uncompressed = nslice*(SUB_HDR + 2*N_fe*N_pe*datasize)
    len_epi2fid = nslice*(N_pe-nseg)*(SUB_HDR + 2*datasize*N_fe)

    f_fid = open(filename,"r")
    f_fid.seek(0,2)
    file_length = f_fid.tell() - MAIN_HDR
    nframe_new = nframe
###print "len_file: %d\nlen_compressed: %d\nlen_uncompressed: %d" % (file_length,nframe*len_compressed,nframe*len_uncompressed)
    f_fid.seek(MAIN_HDR)
 
#   First, assume that the specified number of frames is correct.
    if nframe*len_compressed == file_length:
        fid_format = "compressed"
        print "fid file is in compressed format."
    elif nframe*len_uncompressed == file_length:
        fid_format = "uncompressed"
        print "fid file is in uncompressed format."
    elif nframe*len_epi2fid == file_length:
        print "*** fid file has been converted by epi2fid. ***"
        fid_format = "epi2fid"
        phase_correct = 0
        if time_interp != NONE:
            print "*** Interpolation cannot be done on a fid file in this format. ***"
            time_interp = NONE
    elif nframe*len_asems_ncsnn == file_length:
        fid_format = "asems_ncsnn"
        phase_correct = 0
        time_interp = NONE
        print "fid file is in asems_ncsnn format."
    elif nframe*len_asems_nccnn == file_length:
        fid_format = "asems_nccnn"
        print "fid file is in asems_nccnn format."
    else:
#       That didn't work, so estimate the number of frames for each format and
#       see if it is an integer .
        if file_length % len_compressed == 0:
            fid_format = "compressed"
            nframe_new = file_length/len_compressed
            print "fid file is in compressed format."
        elif file_length % len_uncompressed == 0:
            fid_format = "uncompressed"
            nframe_new = file_length/len_uncompressed
            print file_length,len_uncompressed
            print "fid file is in uncompressed format."
        elif file_length % len_epi2fid == 0:
            print "*** fid file has been converted by epi2fid. ***"
            nframe_new = file_length/len_epi2fid
            fid_format = "epi2fid"
            phase_correct = 0
            if time_interp != NONE:
                print "*** Interpolation cannot be done on a fid file in this format. ***"
                time_interp = NONE
            print "fid file is in epi2fid format."
        elif file_length % len_asems_ncsnn == 0:
            fid_format = "asems_ncsnn"
            nframe_new = file_length/len_asems_ncsnn
            phase_correct = 0
            time_interp = NONE
            print "fid file is in asems_ncsnn format."
        elif file_length % len_asems_nccnn == 0:
            fid_format = "asems_nccnn"
            nframe_new = file_length/len_asems_nccnn
            print "fid file is in asems_nccnn format."
        else:
#           Can'e identify format from its length, try using type of pulse sequence. I 
#           didn't do this first because at some point I didns't trust he correspondence 
#           between this procpar value and the fid format.
            if pulse_sequence == 'epi' or pulse_sequence == 'tepi' or pulse_sequence == 'sparse' or pulse_sequence == 'epidw' or pulse_sequence == 'Vsparse' or pulse_sequence == 'epi_se' or pulse_sequence == epidw_se:
                fid_format = 'compressed'
                nframe_new = file_length/len_compressed 
#                print "file_length % len_compressed: ",file_length % len_compressed
#                print "file_length % len_uncompressed: ",file_length % len_uncompressed
#                print "file_length % len_epi2fid: ",file_length % len_epi2fid
#                print "file_length % len_asems_ncsnn: ",file_length % len_asems_ncsnn
#                print "file_length % len_asems_nccnn: ",file_length % len_asems_nccnn
            else:
                print "*** Error: Cannot recognize fid format. ***"
                return(-1,-1)

    return nframe_new,fid_format

