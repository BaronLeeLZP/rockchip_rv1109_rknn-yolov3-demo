CFLAGS += -I./src
CFLAGS += -I./libopencv_api/include
CFLAGS += -I./librknn_api/include
CFLAGS += -L./libopencv_api/lib
CFLAGS += -L./librknn_api/lib

CXXFLAGS += -I./src
CXXFLAGS += -I./libopencv_api/include
CXXFLAGS += -I./librknn_api/include
CXXFLAGS += -L./libopencv_api/lib
CXXFLAGS += -L./librknn_api/lib

 
LD_OPENCV_LIBS += -lopencv_highgui -lopencv_imgproc -lopencv_dnn -lopencv_imgcodecs  -lopencv_core 
LD_RKNN_LIBS += -lrknn_api

CXXFLAGS += ${LD_OPENCV_LIBS}
CXXFLAGS += ${LD_RKNN_LIBS}
CXXFLAGS += -std=c++11
CXXFLAGS += -Wno-error
CXXFLAGS += -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math  -fpermissive -fpic 
CXXFLAGS += -O3

SRCS1 := $(wildcard ./src/*.c)
SRCS2 := $(wildcard ./src/*.cpp)
TARGET := test

# target source
OBJS1  = $(SRCS1:%.c=%.o)
OBJS2  = $(SRCS2:%.cpp=%.o)

CXX = /home/freebird/arm-rv1109-linux/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++ 
CC = /home/freebird/arm-rv1109-linux/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc
AR = /home/freebird/arm-rv1109-linux/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-ar
.PHONY : clean all

all: $(TARGET)


$(TARGET):   main.o   $(OBJS2) 
	$(CXX) $(CXXFLAGS)  -lpthread -lm -ldl -o $@ $^ -Wl,--start-group ${LD_RKNN_LIBS} -Wl,--end-group
arm_lib:
	$(AR) rcs libyolo.a  $(OBJS2) 
	$(CC) $(CFLAGS) -shared -o libyolo.so   $(OBJS2)   

clean:
	@rm -f $(TARGET) $(OBJS2) ./*.o ./src/*.o
	@rm -f  libyolo.a libyolo.so
	
	
	
	
	
