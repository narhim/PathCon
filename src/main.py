import argparse
from data_loader import load_data
from train import train
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def print_setting(args):
    assert args.use_neighbor or args.use_path
    print()
    print('=============================================')
    print('dataset: ' + args.dataset)
    print('epoch: ' + str(args.epoch))
    print('batch_size: ' + str(args.batch_size))
    print('dim: ' + str(args.dim))
    print('l2: ' + str(args.l2))
    print('lr: ' + str(args.lr))

    print('use local structure message passing: ' + str(args.use_neighbor))
    if args.use_neighbor:
        print('neighbor_samples: ' + str(args.neighbor_samples))
        print('neighbor_hops: ' + str(args.neighbor_hops))
        print('neighbor_agg: ' + args.neighbor_agg)

    print('use entity2entity message passing: ' + str(args.use_path))
    if args.use_path:
        print('max_path_len: ' + str(args.max_path_len))
        print('path mode: ' + args.path_mode)
        if args.path_mode == 'rnn':
            print('path_samples: ' + str(args.path_samples))
            print('path_agg: ' + args.path_agg)
    print('=============================================')
    print()


def main():
    parser = argparse.ArgumentParser()

    '''
    # ===== FB15k ===== #
    parser.add_argument('--dataset', type=str, default='FB15k', help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')

    # settings for local structure message passing
    parser.add_argument('--use_neighbor', type=bool, default=True, help='whether use local structure message passing')
    parser.add_argument('--neighbor_samples', type=int, default=32, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_hops', type=int, default=2, help='number of neighbor hops')
    parser.add_argument('--neighbor_agg', type=str, default='concat', help='neighbor aggregator: mean, concat, cross')

    # settings for entity2entity message passing
    parser.add_argument('--use_path', type=bool, default=True, help='whether use entity2entity message passing')
    parser.add_argument('--max_path_len', type=int, default=2, help='max length of a path')
    parser.add_argument('--path_mode', type=str, default='id', help='path representation mode: id, rnn')
    parser.add_argument('--path_samples', type=int, default=None, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default=None, help='path aggregator if using rnn: mean, att')
    '''

    '''
    # ===== FB15k-237 ===== #
    parser.add_argument('--dataset', type=str, default='FB15k-237', help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')

    # settings for local structure message passing
    parser.add_argument('--use_neighbor', type=bool, default=True, help='whether use local structure message passing')
    parser.add_argument('--neighbor_samples', type=int, default=32, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_hops', type=int, default=2, help='number of neighbor hops')
    parser.add_argument('--neighbor_agg', type=str, default='concat', help='neighbor aggregator: mean, concat, cross')

    # settings for entity2entity message passing
    parser.add_argument('--use_path', type=bool, default=True, help='whether use entity2entity message passing')
    parser.add_argument('--max_path_len', type=int, default=3, help='max length of a path')
    parser.add_argument('--path_mode', type=str, default='id', help='path representation mode: id, rnn')
    parser.add_argument('--path_samples', type=int, default=None, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default=None, help='path aggregator if using rnn: mean, att')
    '''

    '''
    # ===== wn18 ===== #
    parser.add_argument('--dataset', type=str, default='wn18', help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')

    # settings for local structure message passing
    parser.add_argument('--use_neighbor', type=bool, default=True, help='whether use local structure message passing')
    parser.add_argument('--neighbor_samples', type=int, default=16, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_hops', type=int, default=3, help='number of neighbor hops')
    parser.add_argument('--neighbor_agg', type=str, default='cross', help='neighbor aggregator: mean, concat, cross')

    # settings for entity2entity message passing
    parser.add_argument('--use_path', type=bool, default=True, help='whether use entity2entity message passing')
    parser.add_argument('--max_path_len', type=int, default=3, help='max length of a path')
    parser.add_argument('--path_mode', type=str, default='id', help='path representation mode: id, rnn')
    parser.add_argument('--path_samples', type=int, default=None, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default=None, help='path aggregator if using rnn: mean, att')
    '''

    # ===== wn18rr ===== #
    parser.add_argument('--dataset', type=str, default='wn18rr', help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')

    # settings for local structure message passing
    parser.add_argument('--use_neighbor', type=bool, default=True, help='whether use local structure message passing')
    parser.add_argument('--neighbor_samples', type=int, default=8, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_hops', type=int, default=3, help='number of neighbor hops')
    parser.add_argument('--neighbor_agg', type=str, default='cross', help='neighbor aggregator: mean, concat, cross')

    # settings for entity2entity message passing
    parser.add_argument('--use_path', type=bool, default=True, help='whether use entity2entity message passing')
    parser.add_argument('--max_path_len', type=int, default=4, help='max length of a path')
    parser.add_argument('--path_mode', type=str, default='id', help='path representation mode: id, rnn')
    parser.add_argument('--path_samples', type=int, default=None, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default=None, help='path aggregator if using rnn: mean, att')
    
    '''
    # ===== NELL995 ===== #
    parser.add_argument('--dataset', type=str, default='NELL995', help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')

    # settings for local structure message passing
    parser.add_argument('--use_neighbor', type=bool, default=True, help='whether use local structure message passing')
    parser.add_argument('--neighbor_samples', type=int, default=8, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_hops', type=int, default=2, help='number of neighbor hops')
    parser.add_argument('--neighbor_agg', type=str, default='concat', help='neighbor aggregator: mean, concat, cross')

    # settings for entity2entity message passing
    parser.add_argument('--use_path', type=bool, default=True, help='whether use entity2entity message passing')
    parser.add_argument('--max_path_len', type=int, default=3, help='max length of a path')
    parser.add_argument('--path_mode', type=str, default='id', help='path representation mode: id, rnn')
    parser.add_argument('--path_samples', type=int, default=None, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default=None, help='path aggregator if using rnn: mean, att')
    '''

    '''
    # ===== DDB14 ===== #
    parser.add_argument('--dataset', type=str, default='DDB14', help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')

    # settings for local structure message passing
    parser.add_argument('--use_neighbor', type=bool, default=True, help='whether use local structure message passing')
    parser.add_argument('--neighbor_samples', type=int, default=8, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_hops', type=int, default=3, help='number of neighbor hops')
    parser.add_argument('--neighbor_agg', type=str, default='cross', help='neighbor aggregator: mean, concat, cross')

    # settings for entity2entity message passing
    parser.add_argument('--use_path', type=bool, default=True, help='whether use entity2entity message passing')
    parser.add_argument('--max_path_len', type=int, default=4, help='max length of a path')
    parser.add_argument('--path_mode', type=str, default='id', help='path representation mode: id, rnn')
    parser.add_argument('--path_samples', type=int, default=None, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default=None, help='path aggregator if using rnn: mean, att')
    '''

    parser.add_argument('--gpu', type=str, default='0', help='on which GPU to run the code')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print_setting(args)
    data = load_data(args)
    train(args, data)


if __name__ == '__main__':
    main()
