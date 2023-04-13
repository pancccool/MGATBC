import os
os.environ['DGLBACKEND'] = 'tensorflow'
from dgl.nn.tensorflow import *



class MGATBC(tf.keras.Model):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes):
        super(MGATBC, self).__init__()
        self.gat_conv = GATConv(in_dim, hidden_dim, num_heads, attn_drop=0.2, feat_drop=0)
        self.conv2d1 = tf.keras.layers.Conv2D(kernel_size=[1, 120], filters=32, padding='valid')
        self.conv2d2 = tf.keras.layers.Conv2D(kernel_size=[120, 1], filters=64, padding='valid')
        self.mlp1 = tf.keras.layers.Dense(units=96)
        self.classify = tf.keras.layers.Dense(units=n_classes)

    def call(self, bg1, bg2, bg3, bg4):
        h_direct = bg1.ndata['x']
        h_partial = bg2.ndata['x']
        h_spearman = bg3.ndata['x']
        h_knn = bg4.ndata['x']

        # MGAT
        h1 = tf.reshape(self.gat_conv(bg1, h_direct), (bg1.num_nodes(), -1))
        h2 = tf.reshape(self.gat_conv(bg2, h_partial),
                        (bg2.num_nodes(), -1))
        h3 = tf.reshape(self.gat_conv(bg3, h_spearman), (bg3.num_nodes(), -1))
        h4 = tf.reshape(self.gat_conv(bg4, h_knn), (bg4.num_nodes(), -1))

        bg1.ndata['h'] = h1
        bg2.ndata['h'] = h2
        bg3.ndata['h'] = h3
        bg4.ndata['h'] = h4

        batch_feats1 = tf.reshape(h1, (bg1.batch_size, bg1.batch_num_nodes()[0].numpy(), -1))
        batch_feats2 = tf.reshape(h2, (bg2.batch_size, bg2.batch_num_nodes()[0].numpy(), -1))
        batch_feats3 = tf.reshape(h3, (bg3.batch_size, bg3.batch_num_nodes()[0].numpy(), -1))
        batch_feats4 = tf.reshape(h4, (bg4.batch_size, bg4.batch_num_nodes()[0].numpy(), -1))
        # Bilinear
        batch_feats1_bi_pooling = tf.matmul(batch_feats1,
                                            tf.transpose(batch_feats1,
                                                         perm=[0, 2, 1]))
        batch_feats2_bi_pooling = tf.matmul(batch_feats2,
                                            tf.transpose(batch_feats2, perm=[0, 2, 1]))
        batch_feats3_bi_pooling = tf.matmul(batch_feats3,
                                            tf.transpose(batch_feats3,
                                                         perm=[0, 2, 1]))
        batch_feats4_bi_pooling = tf.matmul(batch_feats4,
                                            tf.transpose(batch_feats4, perm=[0, 2, 1]))

        batch_feats1_bi_pooling__ = tf.expand_dims(batch_feats1_bi_pooling, -1)
        batch_feats2_bi_pooling__ = tf.expand_dims(batch_feats2_bi_pooling, -1)
        batch_feats3_bi_pooling__ = tf.expand_dims(batch_feats3_bi_pooling, -1)
        batch_feats4_bi_pooling__ = tf.expand_dims(batch_feats4_bi_pooling, -1)
        # CNN
        conv1_1 = self.conv2d1(batch_feats1_bi_pooling__)
        conv2_1 = self.conv2d1(batch_feats2_bi_pooling__)
        conv3_1 = self.conv2d1(batch_feats3_bi_pooling__)
        conv4_1 = self.conv2d1(batch_feats4_bi_pooling__)

        conv1_2 = self.conv2d2(conv1_1)
        conv2_2 = self.conv2d2(conv2_1)
        conv3_2 = self.conv2d2(conv3_1)
        conv4_2 = self.conv2d2(conv4_1)
        # Connected
        flat_all = tf.reshape(tf.keras.layers.concatenate([conv1_2, conv2_2, conv3_2, conv4_2], -1),
                              (bg1.batch_size, -1))
        # FC
        FC1 = self.mlp1(flat_all)

        return self.classify(FC1)


class MGATBC_re(tf.keras.Model):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes):
        super(MGATBC_re, self).__init__()
        self.gat_conv = GATConv(in_dim, hidden_dim, num_heads, attn_drop=0.2, feat_drop=0)
        self.conv2d1 = tf.keras.layers.Conv2D(kernel_size=[1, 120], filters=32, padding='valid')
        self.conv2d2 = tf.keras.layers.Conv2D(kernel_size=[120, 1], filters=64, padding='valid')
        self.mlp1 = tf.keras.layers.Dense(units=96)
        self.classify = tf.keras.layers.Dense(units=n_classes)

    def call(self, bg1, bg2, bg3, bg4):
        h_direct = bg1.ndata['x']
        h_partial = bg2.ndata['x']
        h_spearman = bg3.ndata['x']
        h_knn = bg4.ndata['x']

        h1 = tf.reshape(self.gat_conv(bg1, h_direct), (bg1.num_nodes(), -1))
        h2 = tf.reshape(self.gat_conv(bg2, h_partial),
                        (bg2.num_nodes(), -1))
        h3 = tf.reshape(self.gat_conv(bg3, h_spearman), (bg3.num_nodes(), -1))
        h4 = tf.reshape(self.gat_conv(bg4, h_knn), (bg4.num_nodes(), -1))

        bg1.ndata['h'] = h1
        bg2.ndata['h'] = h2
        bg3.ndata['h'] = h3
        bg4.ndata['h'] = h4

        batch_feats1 = tf.reshape(h1, (bg1.batch_size, bg1.batch_num_nodes()[0].numpy(), -1))
        batch_feats2 = tf.reshape(h2, (bg2.batch_size, bg2.batch_num_nodes()[0].numpy(), -1))
        batch_feats3 = tf.reshape(h3, (bg3.batch_size, bg3.batch_num_nodes()[0].numpy(), -1))
        batch_feats4 = tf.reshape(h4, (bg4.batch_size, bg4.batch_num_nodes()[0].numpy(), -1))

        batch_feats1_bi_pooling = tf.matmul(batch_feats1,
                                            tf.transpose(batch_feats1,
                                                         perm=[0, 2, 1]))
        batch_feats2_bi_pooling = tf.matmul(batch_feats2,
                                            tf.transpose(batch_feats2, perm=[0, 2, 1]))
        batch_feats3_bi_pooling = tf.matmul(batch_feats3,
                                            tf.transpose(batch_feats3,
                                                         perm=[0, 2, 1]))
        batch_feats4_bi_pooling = tf.matmul(batch_feats4,
                                            tf.transpose(batch_feats4, perm=[0, 2, 1]))


        batch_feats1_bi_pooling__ = tf.expand_dims(batch_feats1_bi_pooling, -1)
        batch_feats2_bi_pooling__ = tf.expand_dims(batch_feats2_bi_pooling, -1)
        batch_feats3_bi_pooling__ = tf.expand_dims(batch_feats3_bi_pooling, -1)
        batch_feats4_bi_pooling__ = tf.expand_dims(batch_feats4_bi_pooling, -1)

        conv1_1 = self.conv2d1(batch_feats1_bi_pooling__)
        conv2_1 = self.conv2d1(batch_feats2_bi_pooling__)
        conv3_1 = self.conv2d1(batch_feats3_bi_pooling__)
        conv4_1 = self.conv2d1(batch_feats4_bi_pooling__)

        conv1_2 = self.conv2d2(conv1_1)
        conv2_2 = self.conv2d2(conv2_1)
        conv3_2 = self.conv2d2(conv3_1)
        conv4_2 = self.conv2d2(conv4_1)

        flat_all = tf.reshape(tf.keras.layers.concatenate([conv1_2, conv2_2, conv3_2, conv4_2], -1),
                              (bg1.batch_size, -1))
        FC1 = self.mlp1(flat_all)
        return self.classify(FC1)