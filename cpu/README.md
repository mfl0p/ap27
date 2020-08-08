# ap26

   Programs for finding large chains of primes in arithmetic progression.

## History

   This code is developed and maintained by Bryan Little

   This is a 10 shift search.  Checks each K with SHIFT and SHIFT+64, SHIFT+128, ..., SHIFT+576

   See http://www.math.uni.wroc.pl/~jwr/AP26/AP26.zip for the original source.

   See http://www.math.uni.wroc.pl/~jwr/AP26/AP26v3.pdf for information
   about how the algorithm works and for his copyleft notice.

## Testing the executable:

   To briefly test the AP26 executable, there are reference output files.
   Check that the file test_x_x_x.txt matches
   the results file SOL-AP26.txt produced by executing:

     AP26.exe x x x

   When testing the OpenCL app, an optional init_data.xml file can be used in
   the directory containing GPU type and device number.
   If no init_data.xml provided, the default device is 0.
   examples:

<app_init_data>
<gpu_type>NVIDIA</gpu_type>
<gpu_device_num>0</gpu_device_num>
</app_init_data>

<app_init_data>
<gpu_type>ATI</gpu_type>
<gpu_device_num>0</gpu_device_num>
</app_init_data>

  The CPU application supports multithreading with the command line -t x
  where x is the number of threads. It cannot exceed the number of logical processors.


## Program operation:

   search parameters are given on the command line as

     AP26.exe [KMIN KMAX SHIFT]

   The search will begin at K=KMIN unless a file AP26-state.txt
   exists containing a checkpoint of the form

     KMIN KMAX SHIFT K checksum

   with KMIN KMAX SHIFT matching the initial search parameters, in which
   case the search will resume from that checkpoint.

   The search will continue up to and including K=KMAX. On completion
   AP26-state.txt will contain a checkpoint of the form

     KMIN KMAX SHIFT KMAX+1 checksum

   Periodic checkpoints will be written to AP26-state.txt.
   All search results and a result checksum will be appended to SOL-AP26.txt.
