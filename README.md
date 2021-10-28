# dotai.trackpingpong.tracknet

开发目的:

面向领域/行业:

主要功能:

技术特点:

## 适配平台

+ MacOS
+ Linux

## 主要外部库引用说明

+ 无

## 主要模块说明

+ 无

## 核心特性说明

+ 无

## 配置说明

+ 无

## 使用说明

+ 无

### 初始化环境

运行 `bin/00_nami_init_for_build.sh`

### Build

+ 普通源代码打包:`bin/01_nami_build_source_distribution.sh`
  + 在本地环境安装:`bin/02_nami_install_from_build.sh`
  + 卸载:`bin/chopper.sh`
+ 普通二进制Wheel打包:`bin/01_nami_build_binary_distribution.sh`
  + 在本地环境安装:`bin/02_nami_install_from_build.sh`
  + 卸载:`bin/chopper.sh`
+ 保护源代码打包:`bin/01_nami_create_secured_source_distribution.sh <relative_path_to_entry_script>`
+ 保护源代码二进制打包:`bin/01_nami_create_secured_pack.sh <relative_path_to_entry_script>`
