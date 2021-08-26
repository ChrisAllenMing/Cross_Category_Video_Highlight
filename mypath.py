class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'YouTube_Highlights':
            root_dir = '~/data/YouTube_Highlights'
            output_dir = '~/data/YouTube_Highlights_processed'
            return root_dir, output_dir
        elif database == 'ActivityNet':
            root_dir = '~/data/ActivityNet'
            output_dir = '~/data/ActivityNet_processed'
            return root_dir, output_dir
        else:
            print('Database {} is not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        pretrained_model_dir = '~/pretrained_models/c3d-pretrained.pth'
        return pretrained_model_dir