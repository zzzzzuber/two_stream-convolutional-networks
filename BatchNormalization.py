import tensorflow as tf

# 实现Batch Normalization
def bn_layer(x,is_training,name='BatchNorm',moving_decay=0.9,eps=1e-5):
    # 获取输入维度并判断是否匹配卷积层(4)或者全连接层(2)
    shape = x.shape
    assert len(shape) in [2,4]

    param_shape = shape[-1]
    with tf.variable_scope(name):
        # 声明BN中唯一需要学习的两个参数，y=gamma*x+beta
        gamma = tf.Variable(tf.ones(shape=param_shape),name='gamma')
        beta  = tf.Variable(tf.zeros(shape=param_shape),name='beat')

        # 计算当前整个batch的均值与方差
        axes = list(range(len(shape)-1))
        batch_mean, batch_var = tf.nn.moments(x,axes,name='moments')

        # 采用滑动平均更新均值与方差
        ema = tf.train.ExponentialMovingAverage(moving_decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean,batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
        mean, var = tf.cond(tf.equal(is_training,True),mean_var_with_update,
                lambda:(ema.average(batch_mean),ema.average(batch_var)))

        # 最后执行batch normalization
        return tf.nn.batch_normalization(x,mean,var,beta,gamma,eps)