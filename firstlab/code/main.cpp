#include "opencv2/core.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/highgui.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <random>
#include <unordered_set>
#include <vector>
#define MAX_LINE 1024
#define FEATURE_COUNT 4
#define CLUSTER_COUNT 3

struct point {
  std::array<float, FEATURE_COUNT> coord = {};
};

struct cluster {
  std::vector<point> points = {};
  point centre;
};

float squared_dist(const point &a, const point &b) {
  float result = 0;
  for (int i = 0; i < FEATURE_COUNT; i++) {
    float di = a.coord[i] - b.coord[i];
    result += di * di;
  }
  return result;
}

cluster *clust_belong(std::array<cluster, CLUSTER_COUNT> &clusters,
                      const point &a) {
  cluster *clust = &clusters[0];
  float nearest = squared_dist(clust->centre, a);
  for (cluster &i : clusters) {
    float temp = squared_dist(i.centre, a);
    if (temp < nearest) {
      nearest = temp;
      clust = &i;
    }
  }
  return clust;
}

point find_centre(const cluster &clust) {
  if (!clust.points.size())
    return {};
  point centre = {};
  for (point i : clust.points) {
    for (int j = 0; j < i.coord.size(); j++) {
      centre.coord[j] += i.coord[j];
    }
  }
  for (float &i : centre.coord) {
    i /= clust.points.size();
  }
  return centre;
}

float clust_centre_dif(const std::array<cluster, CLUSTER_COUNT> &a,
                       const std::array<cluster, CLUSTER_COUNT> &b) {
  float result = 0;
  for (int i = 0; i < CLUSTER_COUNT; i++) {
    for (int j = 0; j < FEATURE_COUNT; j++) {
      result += std::fabs(a[i].centre.coord[j] - b[i].centre.coord[j]);
    }
  }
  return result / (CLUSTER_COUNT * FEATURE_COUNT);
}

std::array<cluster, CLUSTER_COUNT> clustering(std::vector<point> &data,
                                              float E) {

  if (data.size() < CLUSTER_COUNT)
    return {};

  std::array<cluster, CLUSTER_COUNT> current;

  std::mt19937 rng(1234);
  std::uniform_int_distribution<int> dist_int(0, data.size() - 1);

  std::unordered_set<int> chosen;
  int r = 0;

  for (int i = 0; i < current.size(); i++) {
    do {
      r = dist_int(rng);
    } while (chosen.find(r) != chosen.end());
    chosen.insert(r);
    current[i].centre = data[r];
  }

  for (point i : data) {
    clust_belong(current, i)->points.push_back(i);
  }

  std::array<cluster, CLUSTER_COUNT> prev_clust_coord = {};
  do {
    prev_clust_coord = current;
    for (int i = 0; i < CLUSTER_COUNT; i++) {
      current[i].centre = find_centre(current[i]);
    }
    for (cluster &i : current) {
      i.points.clear();
    }
    for (point i : data) {
      clust_belong(current, i)->points.push_back(i);
    }
    // printf("%f\n", clust_centre_dif(prev_clust_coord, current));
  } while (clust_centre_dif(prev_clust_coord, current) > E);

  return current;
}

void print_point(const point &a) {
  for (int i = 0; i < FEATURE_COUNT; i++) {
    printf("\t%f", a.coord[i]);
  }
  printf("\n");
}

void print_cluster(const std::array<cluster, CLUSTER_COUNT> a) {
  for (int i = 0; i < CLUSTER_COUNT; i++) {
    printf("%d cluster countains points:\n", i);
    for (int j = 0; j < a[i].points.size(); j++) {
      print_point(a[i].points[j]);
    }
  }
}

std::vector<point> read_csv_to_points(char *path) {
  std::vector<point> points;
  FILE *fp = fopen(path, "r");
  if (!fp) {
    printf("File reading error\n");
    return {};
  }

  char line[MAX_LINE];

  while (fgets(line, sizeof(line), fp)) {
    char *token = strtok(line, ",");
    point temp = {};
    int count = 0;
    while (token && count < temp.coord.size()) {
      temp.coord[count++] = atof(token);
      token = strtok(NULL, ",");
    }
    points.push_back(temp);
  }
  fclose(fp);
  return points;
}

void write_centres_to_file(std::array<cluster, CLUSTER_COUNT> &a) {
  FILE *fp = fopen("results.csv", "w");
  char line[MAX_LINE] = "";
  for (int i = 0; i < a.size(); i++) {
    for (int j = 0; j < FEATURE_COUNT; j++) {
      char buf[50];
      sprintf(buf, "%f,", a[i].centre.coord[j]);
      strcat(line, buf);
    }
    *strrchr(line, ',') = '\0';
    strcat(line, "\n");
  }
  fprintf(fp, "%s", line);
  fclose(fp);
}

cv::Mat get_graph_base(int img_width, int img_height) {
  cv::Mat image(img_height, img_width, CV_8UC3, cv::Scalar(224, 224, 224));

  if (image.empty()) {
    std::cerr << "Could not create image." << std::endl;
    return {};
  }

  cv::line(image, cv::Point(img_width / 2, 0),
           cv::Point(img_width / 2, img_height), cv::Scalar(0, 0, 0), 1);
  for (int i = 1; i < img_height / 10; i++) {
    cv::line(image, cv::Point(img_width / 2 - 5, 10 * i),
             cv::Point(img_width / 2 + 5, 10 * i), cv::Scalar(0, 0, 0), 1);
  }

  cv::line(image, cv::Point(0, img_height / 2),
           cv::Point(img_width, img_height / 2), cv::Scalar(0, 0, 0), 1);
  for (int i = 1; i < img_height / 10; i++) {
    cv::line(image, cv::Point(10 * i, img_height / 2 + 5),
             cv::Point(10 * i, img_height / 2 - 5), cv::Scalar(0, 0, 0), 1);
  }

  cv::rectangle(image, cv::Point(0, 0),
                cv::Point(img_width - 1, img_height - 1), cv::Scalar(0, 0, 0),
                1);

  return image;
}

cv::Scalar random_color(int idx) {
  static const std::vector<cv::Scalar> baseColors = {
      {255, 0, 0},   {0, 255, 0},   {0, 0, 255},
      {255, 255, 0}, {255, 0, 255}, {0, 255, 255}};
  if (idx < baseColors.size())
    return baseColors[idx];

  int r = (idx * 123) % 256;
  int g = (idx * 321) % 256;
  int b = (idx * 213) % 256;
  return cv::Scalar(b, g, r);
}

float max_coord(std::array<cluster, CLUSTER_COUNT> clusters) {
  float result = 0;
  for (cluster i : clusters) {
    for (point j : i.points) {
      for (float k : j.coord) {
        if (k > result) {
          result = k;
        }
      }
    }
  }
  return result;
}

cv::Mat create_graph(int x, int y, std::array<cluster, CLUSTER_COUNT> clusters,
                     int scale, int dot_scale, int size) {
  if (x == y || x > FEATURE_COUNT - 1 || y > FEATURE_COUNT - 1) {
    return {};
  }

  cv::Mat image = get_graph_base(size * 10 * scale, size * 10 * scale);

  for (int i = 0; i < clusters.size(); i++) {
    for (point j : clusters[i].points) {
      cv::rectangle(image,
                    cv::Point(j.coord[x] * 10 * scale - dot_scale,
                              j.coord[y] * 10 * scale + dot_scale),
                    cv::Point(j.coord[x] * 10 * scale + dot_scale,
                              j.coord[y] * 10 * scale - dot_scale),
                    random_color(i), -1);
    }
  }

  return image;
}

cv::Mat make_border(cv::Mat image) {
  cv::Mat with_border;
  float scale = 0.05;
  cv::copyMakeBorder(image, with_border, image.rows * scale, image.rows * scale,
                     image.cols * scale, image.cols * scale,
                     cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
  return with_border;
}

void print_all(std::array<cluster, CLUSTER_COUNT> clusters, int scale,
               int dot_scale) {
  int size = ceil(max_coord(clusters));
  cv::Mat blank(size * 10 * scale, size * 10 * scale, CV_8UC3,
                cv::Scalar(255, 255, 255));
  blank = make_border(blank);
  int graphs_count = FEATURE_COUNT * (FEATURE_COUNT - 1) / 2;
  int side_size = ceil(sqrt(graphs_count));
  std::vector<cv::Mat> images;
  for (int i = 0; i < FEATURE_COUNT; i++) {
    for (int j = i + 1; j < FEATURE_COUNT; j++) {
      cv::Mat temp = create_graph(i, j, clusters, scale, dot_scale, size);
      images.push_back(make_border(temp));
    }
  }
  cv::Mat image;

  std::vector<cv::Mat> horizontals;
  for (int i = 0; i < ceil((float)graphs_count / side_size); i++) {
    std::vector<cv::Mat> temp;
    for (int j = 0; j < side_size; j++) {
      int temp_coord = i * side_size + j;
      if (temp_coord < graphs_count) {
        temp.push_back(images[temp_coord]);
      } else {
        temp.push_back(blank.clone());
      }
    }
    cv::Mat temp_mat;
    cv::hconcat(temp, temp_mat);
    horizontals.push_back(temp_mat);
  }
  cv::vconcat(horizontals, image);
  cv::imshow("Aboba", image);
  cv::waitKey();
  cv::imwrite("opencv_graphic.png", image);
}

int main() {
  std::vector<point> data = read_csv_to_points("data/Iris_copy.csv");
  std::array<cluster, CLUSTER_COUNT> clusters = clustering(data, 0.1);
  print_cluster(clusters);
  write_centres_to_file(clusters);
  print_all(clusters, 10, 2);
  return 0;
}
