CC = g++
LD = $(CC)

.SUFFIXES:
.SUFFIXES: .o .c .h .cl .cpp

VER = 23_5_6

APP = ap26_ocl_win64_$(VER)

SRC = AP26.cpp simpleCL.c const.h simpleCL.h kernels/checkn.cl kernels/offset.cl kernels/setupok.cl kernels/setupokok.cl kernels/sieve.cl kernels/sieve_nv.cl kernels/setupn.cl kernels/clearn.cl kernels/clearok.cl kernels/clearokok.cl
KERNEL_HEADERS = kernels/checkn.h kernels/offset.h kernels/setupok.h kernels/setupokok.h kernels/sieve.h kernels/sieve_nv.h kernels/setupn.h cl.h kernels/clearn.h kernels/clearok.h kernels/clearokok.h
OBJ = AP26.o simpleCL.o

OCL_LIB = OpenCL.dll

BOINC_DIR = C:/mingwbuilds/boinc
BOINC_INC = -I$(BOINC_DIR)/lib -I$(BOINC_DIR)/api -I$(BOINC_DIR) -I$(BOINC_DIR)/win_build
BOINC_LIB = -L$(BOINC_DIR)/lib -L$(BOINC_DIR)/api -L$(BOINC_DIR) -lboinc_opencl -lboinc_api -lboinc

DFLAGS =
CFLAGS = -I . -I kernels -O3 -m64 -DVERS=\"$(VER)\"
LDFLAGS = $(CFLAGS) -lstdc++ -static

all : clean $(APP) 

$(APP) : $(OBJ)
	$(LD) $(LDFLAGS) $^ $(OCL_LIB) $(BOINC_LIB) -o $@

AP26.o : $(SRC) $(KERNEL_HEADERS)
	$(CC) $(DFLAGS) $(CFLAGS) $(BOINC_INC) -c -o $@ AP26.cpp

simpleCL.o : $(SRC)
	$(CC) $(DFLAGS) $(CFLAGS) $(BOINC_INC) -c -o $@ simpleCL.c

.cl.h:
	perl cltoh.pl $< > $@

clean :
	del *.o
	del kernels\*.h
	del $(APP).exe
	

