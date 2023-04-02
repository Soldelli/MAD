import os
import uuid
import urllib.request
from tqdm import tqdm
from google_measurement_protocol import event, report

class MyProgressBar():
    def __init__(self, filename):
        self.pbar = None
        self.filename = filename

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='iB', unit_scale=True)
            self.pbar.set_description(f"Downloading {self.filename}...")
            self.pbar.refresh()  # to show immediately the update

        self.pbar.update(block_size)


class OwnCloudDownloader():
    def __init__(self, LocalDirectory, OwnCloudServer):
        self.LocalDirectory = LocalDirectory
        self.OwnCloudServer = OwnCloudServer

        self.client_id = uuid.uuid4()

    def downloadFile(self, path_local, path_owncloud, user=None, password=None, verbose=True):
        # return 0: successfully downloaded
        # return 1: HTTPError
        # return 2: unsupported error
        # return 3: file already exist locally
        # return 4: password is None
        # return 5: user is None

        if password is None:
            print(f"password required for {path_local}")
            return 4
        if user is None:
            return 5

        if user is not None or password is not None:  
            # update Password
             
            password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            password_mgr.add_password(
                None, self.OwnCloudServer, user, password)
            handler = urllib.request.HTTPBasicAuthHandler(
                password_mgr)
            opener = urllib.request.build_opener(handler)
            urllib.request.install_opener(opener)

        if os.path.exists(path_local): # check existence
            if verbose:
                print(f"{path_local} already exists")
            return 2

        try:
            try:
                os.makedirs(os.path.dirname(path_local), exist_ok=True)
                urllib.request.urlretrieve(
                    path_owncloud, path_local, MyProgressBar(path_local))

            except urllib.error.HTTPError as identifier:
                print(identifier)
                return 1
        except:
            os.remove(path_local)
            raise
            return 2

        # record googleanalytics event
        data = event('download', os.path.basename(path_owncloud))
        report('UA-99166333-3', self.client_id, data)

        return 0

    
class MadDownloader(OwnCloudDownloader):
    def __init__(self, LocalDirectory, password, download_link, 
                 OwnCloudServer="https://exrcsdrive.kaust.edu.sa/exrcsdrive/public.php/webdav/", ):
        super().__init__(LocalDirectory, OwnCloudServer)
        self.password = password
        self.user = os.path.basename(download_link)

        self.files = [
            'annotations.tar.xz',
            'features/CLIP_B32_frames_features_5fps.h5',
            'features/CLIP_B32_language_features_MAD_test.h5',
            'features/CLIP_B32_language_tokens_features.h5',
            'features/CLIP_L14_frames_features_5fps.h5',
            'features/CLIP_L14_language_tokens_features.h5',
            'DataInspection.ipynb',
        ]

        feat_dir = os.path.join(self.LocalDirectory, 'features')
        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)
        
    def download(self):
        files = '\n\t- '.join(self.files)
        print(f'Downloading files:\n\t- {files}\n')

        for file in self.files:    
            res = self.downloadFile(
                path_local=os.path.join(self.LocalDirectory, file),
                path_owncloud=os.path.join(self.OwnCloudServer, file).replace(' ', '%20').replace('\\', '/'),
                user=self.user,
                password=self.password,
                verbose=True)
        
        self.unpack_tar()
            
    def unpack_tar(self,):
        os.system(f"tar -xf {self.LocalDirectory}/annotations.tar.xz -C {self.LocalDirectory} annotations")
        os.system(f"rm {self.LocalDirectory}/annotations.tar.xz")


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    # Load the arguments
    parser = ArgumentParser(description='MAD Downloader',
                            formatter_class=ArgumentDefaultsHelpFormatter) 

    parser.add_argument('--destination_folder',   required=True,
                        type=str, help='Path to the local destination folder.')
    parser.add_argument('--download_link',   required=True,
                        help = "Link shared by the owner of the data. Apply here: https://forms.gle/hxR4TrQPFuNGpzcr8")
    parser.add_argument('--password',   required=False,
                        type=str, help='Password shared by the owner of the data.')
    args = parser.parse_args()

    Downloader = MadDownloader(
        LocalDirectory=args.destination_folder, 
        password=args.password, 
        download_link=args.download_link
        )
    Downloader.download()
   
