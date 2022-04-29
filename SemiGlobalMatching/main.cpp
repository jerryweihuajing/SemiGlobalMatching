/* -*-c++-*- SemiGlobalMatching - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* Describe	: main
*/

#include "stdafx.h"
#include "SemiGlobalMatching.h"
#include <chrono>
#include <fstream>  

// opencv library
#include <opencv2/opencv.hpp>
//#ifdef _DEBUG
//#pragma comment(lib,"opencv_world310d.lib")
//#else
//#pragma comment(lib,"opencv_world310.lib")
//#endif

using namespace std::chrono;
using namespace std;
using namespace cv;

std::string DATASETS = "RS";
//string DATASETS = "UE4";

#define LEFTSTRIP 0
#define RIGHTSTRIP 1
#define BOTHSTRIP 2

std::string do_strip(const std::string& str, int striptype, const std::string& chars)
{
    std::string::size_type strlen = str.size();
    std::string::size_type charslen = chars.size();
    std::string::size_type i, j;

    //Ĭ������£�ȥ���հ׷�
    if (0 == charslen)
    {
        i = 0;
        //ȥ����߿հ��ַ�
        if (striptype != RIGHTSTRIP)
        {
            while (i < strlen && ::isspace(str[i]))
            {
                i++;
            }
        }
        j = strlen;
        //ȥ���ұ߿հ��ַ�
        if (striptype != LEFTSTRIP)
        {
            j--;
            while (j >= i && ::isspace(str[j]))
            {
                j--;
            }
            j++;
        }
    }
    else
    {
        //��ɾ������תΪc�ַ���
        const char* sep = chars.c_str();
        i = 0;
        if (striptype != RIGHTSTRIP)
        {
            //memchr��������sepָ����ڴ������ǰcharslen���ֽڲ���str[i]
            while (i < strlen && memchr(sep, str[i], charslen))
            {
                i++;
            }
        }
        j = strlen;
        if (striptype != LEFTSTRIP)
        {
            j--;
            while (j >= i && memchr(sep, str[j], charslen))
            {
                j--;
            }
            j++;
        }
        //�������Ҫɾ�����ַ�
        if (0 == i && j == strlen)
        {
            return str;
        }
        else
        {
            return str.substr(i, j - i);
        }
    }

}

std::string strip(const std::string& str, const std::string& chars = " ")
{
    return do_strip(str, BOTHSTRIP, chars);
}

std::string lstrip(const std::string& str, const std::string& chars = " ")
{
    return do_strip(str, LEFTSTRIP, chars);
}

std::string rstrip(const std::string& str, const std::string& chars = " ")
{
    return do_strip(str, RIGHTSTRIP, chars);
}

/*��ʾ�Ӳ�ͼ*/
void ShowDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& name);
/*�����Ӳ�ͼ*/
void SaveDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& path);
/*�����Ӳ����*/
void SaveDisparityCloud(const uint8* img_bytes, const float32* disp_map, const sint32& width, const sint32& height, const std::string& path);

//------------------------------------------------------------------------------
/*
Calculate the path of all the files under the path

Args:
    folder_path: folder path of images

Returns:
    image files
*/
vector<string> VectorFilesPath(string& folder_path) {

    //final result
    vector<string> total_files;

    //File handle for later lookup
    intptr_t hFile = 0;

    //document information
    struct _finddata_t fileinfo;

    //temporary variable
    string path;

    //the first file is found
    if ((hFile = _findfirst(path.assign(folder_path).append("\\*").c_str(), &fileinfo)) != -1) {

        do {
            //condition: folder
            if ((fileinfo.attrib & _A_SUBDIR)) {

                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0) {

                    //files which are from branch folder
                    vector<string> branch_files = VectorFilesPath(path.assign(folder_path).append("\\").append(fileinfo.name));

                    //collect them
                    for (int k = 0; k < branch_files.size(); k++) {

                        total_files.push_back(branch_files[k]);
                    }
                }
            }
            //condition: file
            else {

                total_files.push_back(path.assign(folder_path).append("/").append(fileinfo.name));
            }
        }
        //able to find other files
        while (_findnext(hFile, &fileinfo) == 0);

        //end the lookup and close the handle
        _findclose(hFile);
    }
    return total_files;
}

int SGMFromImageName(std::string folder_path, std::string image_name) {

    cout << image_name << endl;

    std::string image_name_left;
    std::string image_name_right;
    std::string image_name_color;

    if (DATASETS == "UE4") {

        image_name_left = image_name + "_left.bmp";
        image_name_right = image_name + "_right.bmp";
        image_name_color = image_name + "_left.bmp";
    }
    else if (DATASETS == "RS") {

        image_name_left = image_name + "_ir_left.bmp";
        image_name_right = image_name + "_ir_right.bmp";
        image_name_color = image_name + "_color.bmp";
    }
    std::string path_left = folder_path + '/' + image_name_left;
    std::string path_right = folder_path + '/' + image_name_right;
    std::string path_color = folder_path + '/' + image_name_color;

    cv::Mat img_color = cv::imread(path_color, cv::IMREAD_COLOR);
    cv::Mat img_left = cv::imread(path_left, cv::IMREAD_GRAYSCALE);
    cv::Mat img_right = cv::imread(path_right, cv::IMREAD_GRAYSCALE);

    //��������������������������������������������������������������������������������������������������������������������������������������������������������������//
    // ��ȡӰ��
    if (img_left.data == nullptr || img_right.data == nullptr) {
        std::cout << "��ȡӰ��ʧ�ܣ�" << std::endl;
        return -1;
    }
    if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
        std::cout << "����Ӱ��ߴ粻һ�£�" << std::endl;
        return -1;
    }

    //��������������������������������������������������������������������������������������������������������������������������������������������������������������//
    const sint32 width = static_cast<uint32>(img_left.cols);
    const sint32 height = static_cast<uint32>(img_right.rows);

    // ����Ӱ��ĻҶ�����
    auto bytes_left = new uint8[width * height];
    auto bytes_right = new uint8[width * height];

    // ��ͼ�Ĳ�ɫͼ����
    auto bytes_color = new uint8[width * height * 3];

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            bytes_left[i * width + j] = img_left.at<uint8>(i, j);
            bytes_right[i * width + j] = img_right.at<uint8>(i, j);

            bytes_color[i * 3 * width + 3 * j] = img_color.at<cv::Vec3b>(i, j)[0];
            bytes_color[i * 3 * width + 3 * j + 1] = img_color.at<cv::Vec3b>(i, j)[1];
            bytes_color[i * 3 * width + 3 * j + 2] = img_color.at<cv::Vec3b>(i, j)[2];
        }
    }

    printf("Loading Views...Done!\n");

    // SGMƥ��������
    SemiGlobalMatching::SGMOption sgm_option;
    // �ۺ�·����
    sgm_option.num_paths = 8;
    // ��ѡ�ӲΧ
    //sgm_option.min_disparity = argv < 4 ? 0 : atoi(argc[3]);
    //sgm_option.max_disparity = argv < 5 ? 64 : atoi(argc[4]);
    sgm_option.min_disparity = 0;
    sgm_option.max_disparity = 64;
    // census��������
    sgm_option.census_size = SemiGlobalMatching::Census5x5;
    // һ���Լ��
    sgm_option.is_check_lr = true;
    sgm_option.lrcheck_thres = 1.0f;
    // Ψһ��Լ��
    sgm_option.is_check_unique = true;
    sgm_option.uniqueness_ratio = (float)0.99;
    // �޳�С��ͨ��
    sgm_option.is_remove_speckles = true;
    sgm_option.min_speckle_aera = 50;
    // �ͷ���P1��P2
    sgm_option.p1 = 10;
    sgm_option.p2_init = 150;
    // �Ӳ�ͼ���
    // �Ӳ�ͼ���Ľ�������ɿ��������̣���������䣬�����У�������
    sgm_option.is_fill_holes = true;

    printf("w = %d, h = %d, d = [%d,%d]\n\n", width, height, sgm_option.min_disparity, sgm_option.max_disparity);

    // ����SGMƥ����ʵ��
    SemiGlobalMatching sgm;

    //��������������������������������������������������������������������������������������������������������������������������������������������������������������//
    // ��ʼ��
    printf("SGM Initializing...\n");
    auto start = std::chrono::steady_clock::now();
    if (!sgm.Initialize(width, height, sgm_option)) {
        std::cout << "SGM��ʼ��ʧ�ܣ�" << std::endl;
        return -2;
    }
    auto end = std::chrono::steady_clock::now();
    auto tt = duration_cast<std::chrono::milliseconds>(end - start);
    printf("SGM Initializing Done! Timing : %lf s\n\n", tt.count() / 1000.0);

    //��������������������������������������������������������������������������������������������������������������������������������������������������������������//
    // ƥ��
    printf("SGM Matching...\n");
    start = std::chrono::steady_clock::now();
    // disparity���鱣�������ص��Ӳ���
    auto disparity = new float32[uint32(width * height)]();
    if (!sgm.Match(bytes_left, bytes_right, disparity)) {
        std::cout << "SGMƥ��ʧ�ܣ�" << std::endl;
        return -2;
    }
    end = std::chrono::steady_clock::now();
    tt = duration_cast<std::chrono::milliseconds>(end - start);
    printf("\nSGM Matching...Done! Timing :   %lf s\n", tt.count() / 1000.0);

    //��������������������������������������������������������������������������������������������������������������������������������������������������������������//
    // ��ʾ�Ӳ�ͼ
    //ShowDisparityMap(disparity, width, height, "disp-left");
    // �����Ӳ�ͼ
    SaveDisparityMap(disparity, width, height, path_left);
    // �����Ӳ����
    //SaveDisparityCloud(bytes_color, disparity, width, height, path_left);

    cv::waitKey(0);

    //��������������������������������������������������������������������������������������������������������������������������������������������������������������//
    // �ͷ��ڴ�
    delete[] disparity;
    disparity = nullptr;
    delete[] bytes_left;
    bytes_left = nullptr;
    delete[] bytes_right;
    bytes_right = nullptr;
}

void ShowDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& name)
{
    // ��ʾ�Ӳ�ͼ
    const cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
    float32 min_disp = float32(width), max_disp = -float32(width);
    for (sint32 i = 0; i < height; i++) {
        for (sint32 j = 0; j < width; j++) {
            const float32 disp = abs(disp_map[i * width + j]);
            if (disp != Invalid_Float) {
                min_disp = std::min(min_disp, disp);
                max_disp = std::max(max_disp, disp);
            }
        }
    }
    for (sint32 i = 0; i < height; i++) {
        for (sint32 j = 0; j < width; j++) {
            const float32 disp = abs(disp_map[i * width + j]);
            if (disp == Invalid_Float) {
                disp_mat.data[i * width + j] = 0;
            }
            else {
                disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
            }
        }
    }

    cv::imshow(name, disp_mat);
    cv::Mat disp_color;
    applyColorMap(disp_mat, disp_color, cv::COLORMAP_JET);
    cv::imshow(name + "-color", disp_color);

}

void SaveDisparityMap(const float32* disp_map,
    const sint32& width,
    const sint32& height,
    const std::string& path)
{

    std::string temp;
    std::string path_disp_map_tiff;
    std::string path_disp_map_txt;

    if (DATASETS == "RS") {

        temp = path;
        path_disp_map_tiff = temp.replace(temp.find("ir_left.bmp"), 11, "disp_map_SGM.tiff");
       
        temp = path;
        path_disp_map_txt = temp.replace(temp.find("ir_left.bmp"), 11, "disp_map_SGM.txt");
    }
    else if (DATASETS == "UE4") {

        temp = path;
        path_disp_map_tiff = temp.replace(temp.find("left.bmp"), 8, "disp_map_SGM.tiff");

        temp = path;
        path_disp_map_txt = temp.replace(temp.find("left.bmp"), 8, "disp_map_SGM.txt");
    }

    // �����Ӳ�ͼ
    const cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
    float32 min_disp = float32(width), max_disp = -float32(width);
    for (sint32 i = 0; i < height; i++) {
        for (sint32 j = 0; j < width; j++) {
            const float32 disp = abs(disp_map[i * width + j]);
            if (disp != Invalid_Float) {
                min_disp = std::min(min_disp, disp);
                max_disp = std::max(max_disp, disp);
            }
        }
    }
    ofstream fout(path_disp_map_txt);

    for (sint32 i = 0; i < height; i++) {
        for (sint32 j = 0; j < width; j++) {
            const float32 disp = abs(disp_map[i * width + j]);
            if (disp == Invalid_Float) {
                disp_mat.data[i * width + j] = 0;
            }
            else {
                fout << disp << " ";
                disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
            }
        }
        fout << std::endl;
    }
    fout.close();
    cv::imwrite(path_disp_map_tiff, disp_mat);

    //cv::imwrite(path + "-d.png", disp_mat);
    //cv::Mat disp_color;
    //applyColorMap(disp_mat, disp_color, cv::COLORMAP_JET);
    //cv::imwrite(path + "-c.png", disp_color);
    //cv::imwrite(temp.replace(temp.find("left.bmp"), 8, "disp_color_ADC.png"), disp_color);
}


void SaveDisparityCloud(const uint8* img_bytes,
    const float32* disp_map, 
    const sint32& width,
    const sint32& height, 
    const std::string& path)
{
    std::string temp;
    std::string path_disp_cloud_txt;

    if (DATASETS == "RS") {

        temp = path;
        path_disp_cloud_txt = temp.replace(temp.find("ir_left.bmp"), 11, "disp_cloud_SGM.txt");
    }
    else if (DATASETS == "UE4") {

        temp = path;
        path_disp_cloud_txt = temp.replace(temp.find("left.bmp"), 8, "disp_cloud_SGM.txt");
    }

    // �����Ӳ����(x,y,disp,r,g,b)
    FILE* fp_disp_cloud = nullptr;
    if (DATASETS == "RS") {

        fopen_s(&fp_disp_cloud, path_disp_cloud_txt.c_str(), "w");
    }
    else if(DATASETS == "UE4") {

        fopen_s(&fp_disp_cloud, path_disp_cloud_txt.c_str(), "w");
    }
    if (fp_disp_cloud) {
        for (sint32 i = 0; i < height; i++) {
            for (sint32 j = 0; j < width; j++) {
                const float32 disp = abs(disp_map[i * width + j]);
                if (disp == Invalid_Float) {
                    continue;
                }
               /* printf("%f %f %f %d %d %d\n", float32(j), float32(i),
                    disp, img_bytes[i * width * 3 + 3 * j + 2], img_bytes[i * width * 3 + 3 * j + 1], img_bytes[i * width * 3 + 3 * j]);*/

                fprintf_s(fp_disp_cloud, "%f %f %f %d %d %d\n", 
                    float32(j), 
                    float32(i),
                    disp, 
                    img_bytes[i * width * 3 + 3 * j + 2], 
                    img_bytes[i * width * 3 + 3 * j + 1], 
                    img_bytes[i * width * 3 + 3 * j]);
            }
        }
        fclose(fp_disp_cloud);
    }
}

vector<string> split(const string& str, const string& delim) {

    vector<string> res;
    if ("" == str) return res;
    //�Ƚ�Ҫ�и���ַ�����string����ת��Ϊchar*����  
    char* strs = new char[str.length() + 1]; //��Ҫ����  
    strcpy(strs, str.c_str());

    char* d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());

    char* p = strtok(strs, d);
    while (p) {
        string s = p; //�ָ�õ����ַ���ת��Ϊstring����  
        res.push_back(s); //����������  
        p = strtok(NULL, d);
    }

    return res;
}

/**
 * \brief
 * \param argv 3
 * \param argc argc[1]:��Ӱ��·�� argc[2]: ��Ӱ��·�� argc[3]: ��С�Ӳ�[��ѡ��Ĭ��0] argc[4]: ����Ӳ�[��ѡ��Ĭ��64]
 * \param eg. ..\Data\cone\im2.png ..\Data\cone\im6.png 0 64
 * \param eg. ..\Data\Reindeer\view1.png ..\Data\Reindeer\view5.png 0 128
 * \return
 */
int main(int argv, char** argc)
{
    //if (argv < 3) {
    //    std::cout << "�������٣�������ָ������Ӱ��·����" << std::endl;
    //    return -1;
    //}
    //std::string folder_path = "D:/Code/GitHub/3D-Sensors-And-Algorithms-Group/Datasets-Simulation-UE4/Outcome/ArchvizCollectionPackage/datasets";
    std::string folder_path = "D:/Code/GitHub/3D-Sensors/3D-Sensor-Evaluation/Material/RealSense/Static/IMU Recalibrated";

    //std::string image_name = "x-128_y0295_z0180_roll0000_pitch0000_yaw0180";

    //SGMFromImageName(folder_path, image_name);

    for (const auto & item : VectorFilesPath(folder_path)) {

        //��string��Ѱ��sub_string
        if (item.find("left.bmp") != string::npos) {
        
            string this_image_name = strip(split(item, "/").back(), "ir_left.bmp");

            //std::cout << this_image_name << std::endl;
            SGMFromImageName(folder_path, this_image_name);
        }
    }

    system("pause");
    return 0;
}