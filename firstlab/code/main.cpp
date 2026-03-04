#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#define MAX_LINE 1024
#define FEATURE_COUNT 2
#define CLUSTER_COUNT 2
#define GRAPH_SIZE 3000

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

std::vector<point> write_points() {
  std::vector<point> data;
  std::cout << "How many points are you gonna write? -> ";
  int n = 0;
  std::cin >> n;
  while (n <= 0) {
    std::cout << "Wrong input! Try again";
    std::cin >> n;
  }
  std::cout << "Write your points with spaces between each coordinate. Each "
               "point must contain "
            << FEATURE_COUNT << " coordinates!\n";
  for (int i = 0; i < n; i++) {
    std::cout << "-->  ";
    point temp;
    for (int j = 0; j < FEATURE_COUNT; j++) {
      if (!(std::cin >> temp.coord[j])) {
        std::cout << "Wrong input, try again!";
        j--;
      }
    }
    data.push_back(temp);
  }
  return data;
}

void print_point(const point &a) {
  for (int i = 0; i < FEATURE_COUNT; i++) {
    printf("\t%.2f", a.coord[i]);
  }
  printf("\n");
}

void print_all_points(const std::vector<point> &data) {
  for (point i : data) {
    print_point(i);
  }
}

void print_clusters(const std::array<cluster, CLUSTER_COUNT> a) {
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

void write_centres_to_file(std::array<cluster, CLUSTER_COUNT> &a, char *path) {
  FILE *fp = fopen(path, "w");
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

cv::Mat get_graph_base(int size, int division_len) {
  cv::Mat image(size, size, CV_8UC3, cv::Scalar(224, 224, 224));

  float bias = (float)size / 2 - (size / division_len) / 2 * division_len;

  cv::line(image, cv::Point(size / 2, 0), cv::Point(size / 2, size),
           cv::Scalar(0, 0, 0), 1);
  for (int i = 0; i < ceil((float)size / division_len); i++) {
    cv::line(image, cv::Point(size / 2 - 5, division_len * i + bias),
             cv::Point(size / 2 + 5, division_len * i + bias),
             cv::Scalar(0, 0, 0), 1);
  }

  cv::line(image, cv::Point(0, size / 2), cv::Point(size, size / 2),
           cv::Scalar(0, 0, 0), 1);
  for (int i = 0; i < size / division_len; i++) {
    cv::line(image, cv::Point(division_len * i + bias, size / 2 + 5),
             cv::Point(division_len * i + bias, size / 2 - 5),
             cv::Scalar(0, 0, 0), 1);
  }

  cv::rectangle(image, cv::Point(0, 0), cv::Point(size - 1, size - 1),
                cv::Scalar(0, 0, 0), 1);

  return image;
}

cv::Scalar random_color(int idx) {
  static const std::vector<cv::Scalar> baseColors = {
      {200, 30, 30},  {30, 200, 30},  {30, 30, 200},
      {200, 150, 30}, {200, 30, 200}, {30, 200, 200}};
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

cv::Mat create_graph(int x, int y,
                     std::array<cluster, CLUSTER_COUNT> clusters) {
  if (x == y || x > FEATURE_COUNT - 1 || y > FEATURE_COUNT - 1) {
    return {};
  }

  int size = 1000;
  int real_size = max_coord(clusters) + 2;
  int dot_scale = 2;

  cv::Mat image = get_graph_base(size, ceil((float)size / real_size / 2));

  for (int i = 0; i < clusters.size(); i++) {
    for (point j : clusters[i].points) {
      cv::rectangle(
          image,
          cv::Point(
              size / 2. + (j.coord[x] / real_size) * (size / 2.) - dot_scale,
              size / 2. - (j.coord[y] / real_size) * (size / 2.) + dot_scale),
          cv::Point(
              size / 2. + (j.coord[x] / real_size) * (size / 2.) + dot_scale,
              size / 2. - (j.coord[y] / real_size) * (size / 2.) - dot_scale),
          random_color(i), -1);
    }
  }

  return image;
}

cv::Mat make_border(cv::Mat image) {
  cv::Mat with_border;
  float border = 20;
  cv::copyMakeBorder(image, with_border, border, border, border, border,
                     cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
  return with_border;
}

void display_clusters(std::array<cluster, CLUSTER_COUNT> clusters) {
  cv::Mat blank(GRAPH_SIZE, GRAPH_SIZE, CV_8UC3, cv::Scalar(255, 255, 255));
  blank = make_border(blank);
  int graphs_count = FEATURE_COUNT * (FEATURE_COUNT - 1) / 2;
  int side_size = ceil(sqrt(graphs_count));
  std::vector<cv::Mat> images;
  for (int i = 0; i < FEATURE_COUNT; i++) {
    for (int j = i + 1; j < FEATURE_COUNT; j++) {
      cv::Mat temp = create_graph(i, j, clusters);
      temp = make_border(temp);
      cv::putText(temp, std::to_string(i) + ":" + std::to_string(j),
                  cv::Point(20 + 2, 20 - 2), cv::FONT_HERSHEY_COMPLEX, 0.5,
                  cv::Scalar(0, 0, 0), 1);
      images.push_back(temp);
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
}

void display_clusters(std::array<cluster, CLUSTER_COUNT> clusters, char *name) {
  cv::Mat blank(GRAPH_SIZE, GRAPH_SIZE, CV_8UC3, cv::Scalar(255, 255, 255));
  blank = make_border(blank);
  int graphs_count = FEATURE_COUNT * (FEATURE_COUNT - 1) / 2;
  int side_size = ceil(sqrt(graphs_count));
  std::vector<cv::Mat> images;
  for (int i = 0; i < FEATURE_COUNT; i++) {
    for (int j = i + 1; j < FEATURE_COUNT; j++) {
      cv::Mat temp = create_graph(i, j, clusters);
      temp = make_border(temp);
      cv::putText(temp, std::to_string(i) + ":" + std::to_string(j),
                  cv::Point(20 + 2, 20 - 2), cv::FONT_HERSHEY_COMPLEX, 0.5,
                  cv::Scalar(0, 0, 0), 1);
      images.push_back(temp);
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
  cv::imwrite(name, image);
}

void menu(std::vector<point> &data) {
  std::array<cluster, CLUSTER_COUNT> clusters;
  char path[64];
  while (true) {
    system("clear");
    printf("============\n=   Menu   =\n============\n\n0) Exit\n1) Print "
           "current points\n2) Write points by "
           "hand\n3) Read points from text file\n4) Cluster current "
           "points\n5) Pirnt current cluster partitioning\n6) Write centres "
           "of clusters to a file\n7) Save an image of clusters\n8) Display a "
           "window with clusters (doesn't work properly)\n\nAction -> ");
    int choice = 0;
    scanf("%d", &choice);
    getchar();
    while (0 > choice && 7 < choice) {
      printf("Wrong input! Try again.");
      scanf("%d", &choice);
      getchar();
      continue;
    }
    switch (choice) {
    case 0:
      return;
      break;
    case 1:
      print_all_points(data);
      getchar();
      break;
    case 2:
      data = write_points();
      break;
    case 3:
      printf("Give a path to your file -> ");
      fgets(path, sizeof(path), stdin);
      if (strchr(path, '\n')) {
        *strchr(path, '\n') = 0;
      }
      while (strlen(path) > 50) {
        printf("The name is too long! Try again.");
        fgets(path, sizeof(path), stdin);
        if (strchr(path, '\n')) {
          *strchr(path, '\n') = 0;
        }
      }
      data = read_csv_to_points(path);
      break;
    case 4:
      clusters = clustering(data, 10E-3);
      break;
    case 5:
      print_clusters(clusters);
      getchar();
      break;
    case 6:
      printf("Give a name to your file -> ");
      fgets(path, sizeof(path), stdin);
      if (strchr(path, '\n')) {
        *strchr(path, '\n') = 0;
      }
      while (strlen(path) > 50) {
        printf("The name is too long! Try again.");
        fgets(path, sizeof(path), stdin);
        if (strchr(path, '\n')) {
          *strchr(path, '\n') = 0;
        }
      }
      write_centres_to_file(clusters, path);
      break;
    case 7:
      printf("Give a name to your file -> ");
      fgets(path, sizeof(path), stdin);
      if (strchr(path, '\n')) {
        *strchr(path, '\n') = 0;
      }
      while (strlen(path) > 50) {
        printf("The name is too long! Try again.");
        fgets(path, sizeof(path), stdin);
        if (strchr(path, '\n')) {
          *strchr(path, '\n') = 0;
        }
      }
      display_clusters(clusters, path);
      break;
    case 8:
      display_clusters(clusters);
    }
  }
}

int main() {
  std::vector<point> data;
  menu(data);
  return 0;
}
