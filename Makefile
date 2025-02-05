# make xxx 默认优先执行命令运行目录中的 xxx 文件
# 使用 .PHONY 标识执行 make clean 等命令时，如果当前目录中存在以下字符同名的文件，
# 会忽略这些文件，而优先执行如下定义的构建过程
.PHONY : build clean format install-python test-cpp test-onnx

TYPE ?= Release
TEST ?= ON

CMAKE_OPT = -DCMAKE_BUILD_TYPE=$(TYPE)
CMAKE_OPT += -DBUILD_TEST=$(TEST)

# 没有指定默认目标时，直接执行 make 命令会运行定义的第一个目标
build:
	mkdir -p build/$(TYPE)
	cd build/$(TYPE) && cmake $(CMAKE_OPT) ../.. && make -j8

clean:
	rm -rf build

test-cpp:
	@echo
	cd build/$(TYPE) && make test
