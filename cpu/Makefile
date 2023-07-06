CC = g++
LD = $(CC)

VER = 23_7_5

APP = ap26_cpu_win64_$(VER)

SRC = AP26.cpp
OBJ = AP26.o cpuavx512.o cpuavx2.o cpuavx.o cpusse41.o cpusse2.o

BOINC_DIR = C:/mingwbuilds/boinc
BOINC_INC = -I$(BOINC_DIR)/lib -I$(BOINC_DIR)/api -I$(BOINC_DIR) -I$(BOINC_DIR)/win_build
BOINC_LIB = -L$(BOINC_DIR)/lib -L$(BOINC_DIR)/api -L$(BOINC_DIR) -lboinc_api -lboinc -lstdc++ -lpthread

DFLAGS =
CFLAGS = -I . -O3 -m64 -DVERS=\"$(VER)\"
LDFLAGS = $(CFLAGS) -static

all : clean $(APP)

$(APP) : $(OBJ)
	$(LD) $(LDFLAGS) $^ $(BOINC_LIB) -o $@ 

AP26.o : $(SRC)
	$(CC) $(DFLAGS) $(CFLAGS) $(BOINC_INC) -c -o $@ AP26.cpp

cpuavx512.o : cpuavx512.cpp
	$(CC) $(DFLAGS) $(CFLAGS) -mavx512bw -mavx512vl -c -o $@ $^

cpuavx2.o : cpuavx2.cpp
	$(CC) $(DFLAGS) $(CFLAGS) -mavx2 -c -o $@ $^

cpuavx.o : cpuavx.cpp
	$(CC) $(DFLAGS) $(CFLAGS) -mavx -c -o $@ $^

cpusse41.o : cpusse41.cpp
	$(CC) $(DFLAGS) $(CFLAGS) -msse4.1 -c -o $@ $^

cpusse2.o : cpusse2.cpp
	$(CC) $(DFLAGS) $(CFLAGS) -msse2 -c -o $@ $^

clean :
	del *.o
	del $(APP).exe

