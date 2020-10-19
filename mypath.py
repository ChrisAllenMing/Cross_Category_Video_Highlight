class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = '/Path/to/UCF-101'

            # Save preprocess data into output_dir
            output_dir = '/path/to/VAR/ucf101'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/Path/to/hmdb-51'

            output_dir = '/path/to/VAR/hmdb51'

            return root_dir, output_dir
        elif database == 'YouTube_Highlights':
            # the direction to the raw data
            # root_dir = './datasets/YouTube_Highlights'
            # root_dir = '/Users/admin/Desktop/datasets/YouTube_Highlights'
            root_dir = 'hdfs://haruna/home/byte_arnold_hl_vc/xuminghao.118/data/video_highlights/YouTube_Highlights'

            # the direction to the preprocessed data
            # output_dir = './datasets/YouTube_Highlights_processed'
            # output_dir = '/Users/admin/Desktop/datasets/YouTube_Highlights_processed'
            output_dir = 'hdfs://haruna/home/byte_arnold_hl_vc/xuminghao.118/data/video_highlights/YouTube_Highlights_processed'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        # return './pretrained_model/c3d-pretrained.pth'
        return 'hdfs://haruna/home/byte_arnold_hl_vc/xuminghao.118/pretrained_models/c3d-pretrained.pth'