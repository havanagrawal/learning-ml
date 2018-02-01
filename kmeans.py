from collections import defaultdict
import numpy as np


def euclidean_distance(r1, r2):
    r1, r2 = np.array(r1), np.array(r2)
    return np.sqrt(np.sum((r1 - r2)**2))


def _kmeans_clusters_init(data, k, distance_metric):
    n = len(data)
    seeds = np.random.randint(n, size=k)
    seed_points = [data[seed] for seed in seeds]

    return _move_clusters_to_new_means(data, distance_metric, seed_points)


def _move_clusters_to_new_means(data, distance_metric, means):
    clusters = defaultdict(list)

    for row in data:
        min_distance = np.inf
        closest_centroid = None
        for i, centroid in enumerate(means):
            d = distance_metric(centroid, row)
            if d < min_distance:
                closest_centroid = i
                min_distance = d

        clusters[closest_centroid].append(row)

    return clusters


def _get_cluster_means(clusters):
    return [np.mean(cluster) for seed, cluster in clusters.items()]


def kmeans_clusters(data, k, distance_metric, iterations=3):
    clusters = _kmeans_clusters_init(data, k, distance_metric)

    for _ in range(iterations):
        cluster_means = _get_cluster_means(clusters)
        clusters = _move_clusters_to_new_means(data, distance_metric, cluster_means)

    return clusters


def main():
    sample_data = [
        [1, 1], [1, 2], [0, 1], [1, 0], [1, 2], [2, 1], [0, 0],
        [10, 1], [10, 2], [10, 1], [11, 0], [11, 2], [12, 1], [10, 0],
    ]
    c = kmeans_clusters(sample_data, 2, euclidean_distance, iterations=5)
    print(c)

if __name__ == "__main__":
    main()
