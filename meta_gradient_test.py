import dataset
import tensorflow as tf
import grouplearner
import optimizer as op
import spec
import numpy as np

def main(argv):
    #n_test = argv[1]
    learning_rate = 5e-3
    n_epoch = 1
    n_batch = 10
    n_task = 10
    learning_rates = learning_rate * np.ones(n_task)
    learning_specs = []

    # model path
    model_dir = "meta_test"
    np.random.seed(0)

    # config
    run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=6000)
    ws0 = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=model_dir, vars_to_warm_start=["meta"])
    ws1 = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=model_dir, vars_to_warm_start=["main", "meta"])

    # generate sequence dataset
    set_of_datasets = dataset.SetOfRandPermMnist(n_task)
    d_in = set_of_datasets.list[0].d_in

    # learning specs
    for i in range(n_task):
        opt = op.SGDOptimizer().build(learning_rates[i])
        opt_spec = spec.OptimizerSpec(opt, d_in)
        learning_specs.append(spec.LearningSpec(n_epoch, n_batch, n_task, model_dir, opt_spec))

    my_grouplearner = grouplearner.GroupGradientMetaTestLearner(set_of_datasets, learning_specs, n_task, run_config, ws0, ws1)

    accuracy_matrix = my_grouplearner.train_and_evaluate()

    n_total = n_task * (n_task + 1) / 2.0
    average_accuracy = accuracy_matrix.sum() / n_total

    print(accuracy_matrix)
    print(average_accuracy)

    # filepath = "result/meta_2/" + str(n_test)
    # f = open(filepath, 'w')
    # f.write(str(average_accuracy))
    # f.close()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
