def save(filepath, model_dir, accuracy_matrix, metric_list, seed, learning_specs, step=0, n_grid=0):

    alpha = learning_specs[0].alpha
    f = open(filepath, 'a')
    f.write(model_dir)
    f.write("(seed, alpha, step, grid) = (" + str(seed) + "," + str(alpha) + "," + str(step) + "," + str(n_grid) + ") \n")

    x, y = accuracy_matrix.shape

    save_matrix(accuracy_matrix, x, y, f)
    save_metrics(metric_list, f)

    f.close()


def save_vector(accuracy_vector, y, f):
    for j in range(y):
        f.write(str(round(accuracy_vector[j], 4)) + " ")
    f.write(str("\n"))


def save_matrix(accuracy_matrix, x, y, f):
    for i in range(x):
        save_vector(accuracy_matrix[i], y, f)
    f.write(str("\n"))


def save_metrics(metric_list, f):
    for item in metric_list:
        f.write(str(round(item, 4)) + "\n")


