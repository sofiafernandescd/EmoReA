emotions_dict = {
        'empty': ['empty', 'void', 'hollow'],
        'sadness': ['sad', 'melancholy', 'depressed'],
        'enthusiasm': ['enthusiastic', 'excited', 'eager'],
        'neutral': ['neutral', 'indifferent', 'unbiased'],
        'worry': ['worry', 'anxiety', 'concern'],
        'surprise': ['surprise', 'astonishment', 'shock'],
        'love': ['love', 'affection', 'adoration'],
        'fun': ['fun', 'joyful', 'amusing'],
        'hate': ['hate', 'detest', 'loathe'],
        'happiness': ['happy', 'joy', 'content'],
        'boredom': ['boredom', 'tedium', 'monotony'],
        'relief': ['relief', 'ease', 'comfort'],
        'anger': ['angry', 'rage', 'outrage']
    }

import pandas as pd
import os, sys
import glob
from kagglehub import kagglehub
import pickle
import numpy as np
import cv2
from tqdm import tqdm
import json
module_path = os.path.abspath(os.path.join('..', '..')) # or the path to your source code
sys.path.insert(0, module_path)

def load_csd_file(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def filter_emotions(df):
    """ Filters the DataFrame to include only selected emotions.
        The seven selected emotions are the Ekman's six basic emotions 
        (happy, sad, angry, fear, disgust, and surprise) + neutral.
    Args:
        df (pd.DataFrame): DataFrame containing emotion data with a column 'emotion'.
    Returns:
        pd.DataFrame: Filtered DataFrame with only selected emotions.
    Raises:
        ValueError: If 'emotion' column is not present in the DataFrame.
    """    
    # Check if 'label' column exists
    if 'label' not in df.columns:
        raise ValueError("DataFrame must contain an 'label' column.")
    # filter selected emotions
    selected_emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
    df = df[df['label'].isin(selected_emotions)]
    # reset index
    df.reset_index(drop=True, inplace=True)
    return df

######### TER Datasets #########
def load_emorynlp(split='train'):
    emory_path = os.path.join(module_path, 'data', 'EmoryNLP', f'emorynlp_{split}.json')
    with open(emory_path, 'r') as f:
        emory_data = json.load(f)['episodes']
    print(f"Loaded {len(emory_data)} episodes from EmoryNLP {split} dataset")

    texts, labels = [], []
    for episode in emory_data:
        for scene in episode['scenes']:
            for utt in scene['utterances']:
                texts.append(utt['transcript'])
                labels.append(utt['emotion'].lower())

    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    emo_map = {
        'joyful': 'happy',
        'scared': 'fear',
        'neutral': 'neutral',
        'mad' : 'mad',
        'peaceful': 'peaceful',
        'powerful':'powerful',
        'sad':'sad'

    }
    df['label'] = df['label'].map(emo_map)
    return df

def load_isear():

    # download from Kaggle
    path = kagglehub.dataset_download("faisalsanto007/isear-dataset")
    path_data = os.path.join(path, "eng_dataset.csv")

    # rename columns
    df = pd.read_csv(path_data)
    df.columns = ['id', 'label', 'text']

    emo_map = {
        'joy': 'happy',
        'sadness': 'sad',
        'anger': 'angry',
        'fear': 'fear'
    }    
    df['label'] = df['label'].map(emo_map)
    print(f"Loaded {len(df)} samples from ISEAR dataset")
    return df



######### SER Datasets #########
def load_emodb():
    """
    The EMODB (Emotional Database) is a German emotional speech database that contains recordings of actors reading sentences with different emotions. It is widely used for emotion recognition tasks in speech processing.
    The EMODB database comprises of seven emotions: 1) anger; 2) boredom; 3) anxiety; 4) happiness; 5) sadness; 6) disgust; and 7) neutral. 
    The data was recorded at a 48-kHz sampling rate and then down-sampled to 16-kHz.
    Every utterance is named according to the same scheme.
    Example: 03a01Fa.wav is the audio file from Speaker 03 speaking text a01 with the emotion "Freude" (Happiness).
    From: https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb
    Returns:
        pd.DataFrame: DataFrame with columns ['filename', 'label']
    """
    # Download dataset from Kaggle
    path = kagglehub.dataset_download("piyushagni5/berlin-database-of-emotional-speech-emodb")
    print("Path to dataset files:", path)

    # List all files in the dataset
    files = [file_path.name for file_path in os.scandir(path+"/wav") if file_path.is_file()]

    # Define the dataset directory
    emodb_path = path + "/wav/*.wav"

    # Codebook that maps EmoDB filename encoding to emotions
    emotion_map = {
        'W': 'angry', #anger
        'L': 'boredom',
        'E': 'disgust',
        'A': 'fear',
        'F': 'happy', #hapiness
        'T': 'sad', #sadness
        'N': 'neutral'
    }

    # Extract emotion labels from filenames (6th character)
    filenames = glob.glob(emodb_path)
    emotions = [emotion_map[os.path.basename(f)[5]] for f in filenames if os.path.basename(f)[5] in emotion_map]

    # Create a DataFrame to store the emotion labels
    emo_db = pd.DataFrame({
        'filename': filenames,
        'label': emotions
    })

    return emo_db


def load_tess():
    """
    The TESS (Toronto emotional speech set) is a dataset that contains audio recordings of actors reading sentences with different emotions. 
    It includes a wide range of emotions such as happiness, sadness, anger, fear, surprise, and disgust.
    The dataset is widely used for emotion recognition tasks in speech processing.
    Returns:
        pd.DataFrame: DataFrame with columns ['filename', 'label']
    """
    # Download dataset from Kaggle
    path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")
    print("Path to dataset files:", path)
    # List all files in the dataset
    print(os.listdir(path))
    # change name of existing folder on path
    os.rename(os.path.join(path, os.listdir(path)[0]), os.path.join(path, "TESS"))

    tess_path = os.path.join(path, "TESS")
    print(os.listdir(tess_path))

    all_files = [
        os.path.join(tess_path, folder, filename)
        for folder in os.listdir(tess_path)
        for filename in os.listdir(os.path.join(tess_path, folder))
    ]

    labels = [
        f[:-4].split('_')[-1].lower() 
        if f[:-4].split('_')[-1].lower() != "ps"
        else "surprise"
        for f in all_files
    ]
    # Create a DataFrame to store the emotion labels
    tess_db = pd.DataFrame({
        'filename': all_files,
        'label': labels
    })

    return tess_db


def load_crema_d():
    """
    The CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset) is a dataset that contains audio and video recordings of actors performing emotional speech. 
    It includes a wide range of emotions such as happiness, sadness, anger, fear, surprise, and disgust.
    The dataset is widely used for emotion recognition tasks in both audio and visual modalities.
    Returns:
        pd.DataFrame: DataFrame with columns ['filename', 'label']
    """
    # Download dataset from Kaggle
    path = kagglehub.dataset_download("ejlok1/cremad")
    print("Path to dataset files:", path)
    # List all files in the dataset
    path = path + '/AudioWAV'
    crema_directory_list = os.listdir(path)
    print(crema_directory_list)

    file_emotion = []
    file_path = []

    for file in crema_directory_list:
        # storing file paths
        file_path.append(path + '/' + file)
        # storing file emotions
        part=file.split('_')
        if part[2] == 'SAD':
            file_emotion.append('sad')
        elif part[2] == 'ANG':
            file_emotion.append('angry')
        elif part[2] == 'DIS':
            file_emotion.append('disgust')
        elif part[2] == 'FEA':
            file_emotion.append('fear')
        elif part[2] == 'HAP':
            file_emotion.append('happy')
        elif part[2] == 'NEU':
            file_emotion.append('neutral')
        else:
            file_emotion.append('unknown')
            
    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['label'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['filename'])
    crema_df = pd.concat([emotion_df, path_df], axis=1)

    return crema_df

def load_ravdess():
    """
    The RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) is a dataset that contains audio and video recordings of actors performing emotional speech and song. 
    It includes a wide range of emotions such as happiness, sadness, anger, fear, surprise, and disgust.
    The dataset is widely used for emotion recognition tasks in both audio and visual modalities.
    Returns:
        pd.DataFrame: DataFrame with columns ['filename', 'label']
    """
    # Download dataset from Kaggle
    path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
    print("Path to dataset files:", path)
    # List all files in the dataset
    print(os.listdir(path))

    # Define the dataset directory
    ravdess_path = path + "/audio_speech_actors_01-24/**/*.wav"

    # Codebook that maps RAVDESS filename encoding to emotions
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fear', #'fearful',
        '07': 'disgust',
        '08': 'surprise' #'surprised'
    }

    # Extract emotion labels from filenames
    filenames = glob.glob(ravdess_path, recursive=True)
    emotions = [emotion_map[os.path.basename(f).split('-')[2]] for f in filenames if os.path.basename(f).split('-')[2] in emotion_map]

    # Create a DataFrame to store the emotion labels
    ravdess_db = pd.DataFrame({
        'filename': filenames,
        'label': emotions
    })

    return ravdess_db


def load_savee():
    pass


######### FER Datasets #########
def load_ck():
    """
    The CK+ (Cohn-Kanade) dataset is a widely used dataset for facial expression recognition. 
    It contains video sequences of facial expressions from a diverse set of subjects, annotated with emotion labels.
    The dataset is commonly used for training and evaluating models in facial expression recognition tasks.
    Returns:
        pd.DataFrame: DataFrame with columns ['filename', 'label']

    """
    # Download dataset
    file_path = kagglehub.dataset_download("davilsena/ckdataset")
    print("Subfolders in the dataset:", os.listdir(file_path))

    df = pd.read_csv(os.path.join(file_path, "ckextended.csv"))
    
    # map emotion labels
    emotion_map = {
        0: "angry", #"anger",
        1: "disgust",
        2: "fear",
        3: "happy", #"happiness",
        4: "sad", #"sadness",
        5: "surprise",
        6: "neutral",
        7: "contempt"
    }
        
    df['label'] = df['emotion'].map(emotion_map)
    return df

def load_fer2013():
    """
    The FER2013 (Facial Expression Recognition 2013) dataset is a widely used dataset for facial expression recognition tasks. 
    It contains grayscale images of faces with corresponding emotion labels, including happiness, sadness, anger, fear, surprise, and disgust.
    The dataset is commonly used for training and evaluating models in facial expression recognition tasks.
    Returns:
        pd.DataFrame: DataFrame with columns ['filename', 'label']
    """
    # download dataset
    file_path = kagglehub.dataset_download("deadskull7/fer2013")
    print("Subfolders in the dataset:", os.listdir(file_path))
    # read dataset
    df_fer = pd.read_csv(os.path.join(file_path, "fer2013.csv"))
    print("First few rows of the dataframe:")
    print(df_fer.head())

    
    
    df_fer['label'] = df_fer['emotion'].map(label_to_text)
    
    return df_fer

def load_rafbd(folder="test"):
    # Download the dataset
    file_path = kagglehub.dataset_download("shuvoalok/raf-db-dataset")
    imgs_path = os.path.join(file_path, "DATASET", folder)
    # Check subfolders
    print("Subfolders:", os.listdir(file_path))
    print("Subfolders in the dataset:", os.listdir(imgs_path))
    # read test labels
    df_raf = pd.read_csv(os.path.join(file_path, f"{folder}_labels.csv"))
    emotion_map = {
        1: "surprise",
        2: "fear",
        3: "disgust",
        4: "happy",
        5: "sad",
        6: "angry",
        7: "neutral"
    }

    df_raf['emo'] = df_raf['label']
    df_raf['label'] = df_raf['emo'].map(emotion_map)
    raf_imgs = []
    # read images from each of the folders in column "label"
    for i in tqdm(range(len(df_raf))):
        folder = str(df_raf.iloc[i]['emo'])
        img_path = os.path.join(imgs_path, folder, df_raf.iloc[i]['image'])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting
                raf_imgs.append(img_rgb)
            else:
                raf_imgs.append("Missing image")

    df_raf['pixels'] = raf_imgs
    return df_raf

def load_affectnet():
    
    data_path = kagglehub.dataset_download('mstjebashazida/affectnet')
    print(f"Dataset downloaded to: {data_path}")
    list_folders = os.listdir(os.path.join(data_path, 'archive (3)', 'Test'))
    print(f"Available folders: {list_folders}")
    filenames, labels, imgs = [], [], []
    for emo in tqdm(list_folders):
        if emo!="Contempt":
            files = os.path.join(data_path, 'archive (3)', 'Test', emo)
            for file in os.listdir(files):
                if emo=="Anger":
                    labels.append("angry")
                else:
                    labels.append(emo.lower())
                filenames.append(os.path.join(files, file))
                imgs.append(cv2.imread(os.path.join(files, file)))

    df = pd.DataFrame({
        'filename': filenames,
        'label': labels,
        'pixels': imgs
    })

    return df

###### MER DATASETS #######
def load_iemocap():
    """
    The IEMOCAP (Interactive Emotional Motion Capture) dataset is a multimodal dataset that contains audio, video, and motion capture recordings of actors performing emotional dialogues. 
    It includes a wide range of emotions such as happiness, sadness, anger, fear, surprise, and disgust.
    The dataset is widely used for training and evaluating models in multimodal emotion recognition tasks.
    Returns:
        pd.DataFrame: DataFrame with columns ['filename', 'label']
    """
    # Download dataset
    file_path = kagglehub.dataset_download("dejolilandry/iemocapfullrelease")
    # Check subfolders
    print("Subfolders in the dataset:", os.listdir(file_path))

    labels_path = kagglehub.dataset_download("samuelsamsudinng/iemocap-emotion-speech-database")
    labels_path = os.path.join(labels_path, "iemocap_full_dataset.csv")
    # Load the labels CSV file
    labels_df = pd.read_csv(labels_path)
    #allowed_emotions = ['neu', 'sad', 'ang', 'hap', 'sur', 'fea', 'dis']
    #df = labels_df[labels_df['label'].isin(allowed_emotions)].copy()#.sample(100)
    emos = {
        'neu': 'neutral',
        'sad': 'sad',
        'ang': 'angry',
        'hap': 'happy',
        'sur': 'surprise',
        'fea': 'fear',
        'dis': 'disgust'
    }
    df = labels_df.copy()
    df['label'] = df['emotion'].map(emos)
    df['filename'] = df['path'].apply(lambda x: os.path.join(file_path, "IEMOCAP_full_release", x))
        
    return df


def load_meld():
    """
    The MELD (Multimodal EmotionLines Dataset) is a dataset that contains audio, video, and text data from conversations in TV series. 
    It includes a wide range of emotions such as happiness, sadness, anger, fear, surprise, and disgust.
    The dataset is widely used for training and evaluating models in multimodal emotion recognition tasks.
    Returns:
        pd.DataFrame: DataFrame with columns ['filename', 'label']
    """
    # Download dataset
    file_path = kagglehub.dataset_download("bhandariprakanda/meld-emotion-recognition")
    # Check subfolders
    print("Subfolders in the dataset:", os.listdir(file_path))
    raw_path = os.path.join(file_path, "MELD.Raw", "MELD.Raw", "test", "output_repeated_splits_test")
    files = os.listdir(raw_path)
    # scan the directory 
    print("Subfolders in the raw data:", files)
    # create dataframe with the files
    filenames = [os.path.join(raw_path, file) for file in files]
    # Load the labels CSV file
    labels_path = os.path.join(file_path, "MELD.Raw", "MELD.Raw", "test_sent_emo.csv")
    labels_df = pd.read_csv(labels_path)
    labels_df['filename'] = labels_df[['Dialogue_ID', 'Utterance_ID']].apply(
        lambda x: os.path.join(raw_path, f"dia{x['Dialogue_ID']}_utt{x['Utterance_ID']}.mp4"), axis=1
        )
    emo_map = {
        'neutral': 'neutral',
        'joy': 'happy',
        'anger': 'angry',
        'surprise': 'surprise',
        'sadness': 'sad',
        'disgust': 'disgust',
        'fear': 'fear'
    }
    labels_df['label'] = labels_df['Emotion'].map(emo_map)
    
    return labels_df

def load_cmu_mosi():
    """
    The CMU-MOSEI (CMU Multimodal Opinion Sentiment and Emotion Intensity) dataset is a large-scale dataset for multimodal sentiment analysis and emotion recognition. 
    It contains video clips of people expressing various emotions, along with corresponding text transcriptions and audio recordings.
    The dataset is widely used for training and evaluating models in multimodal emotion recognition tasks.
    Every segment in the dataset is represented as a dictionary with the following structure:
    {
    'features': List[np.ndarray],  # Lista de arrays de features por segmento
    'intervals': List[Tuple[float, float]],  # Início e fim dos segmentos
    'timestamps': List[str],  # Nome dos segmentos (ex: 'vid1234[0]')
    'metadata': Dict  # Info extra como dimension, sampling rate, etc.
    }
    Returns:
        pd.DataFrame: DataFrame with columns ['filename', 'label']
    """
    # Download dataset from Kaggle
    path = kagglehub.dataset_download("samarwarsi/cmu-mosei")
    print("Path to dataset files:", path)
    # List all files in the dataset
    print(os.listdir(path))

    covarep = load_csd_file("cmu-mosei/CMU_MOSEI_COVAREP.csd")
    facet = load_csd_file("cmu-mosei/CMU_MOSEI_VisualFacet_2.3.csd")
    glove = load_csd_file("cmu-mosei/CMU_MOSEI_TimestampedWords.csd")
    labels = load_csd_file("cmu-mosei/CMU_MOSEI_Opinion_Labels.csd")

    print("Exemplo de segmento:", labels['timestamps'][0])
    print("Label:", labels['features'][0])
    print("Áudio (COVAREP):", covarep['features'][0].shape)
    print("Vídeo (FACET):", facet['features'][0].shape)

    data = []
    for i, seg_id in enumerate(labels['timestamps']):
        row = {
            'segment_id': seg_id,
            'label': labels['features'][i],
            'covarep': np.mean(covarep['features'][i], axis=0) if seg_id in covarep['timestamps'] else None,
            'facet': np.mean(facet['features'][i], axis=0) if seg_id in facet['timestamps'] else None,
            'glove': np.mean(glove['features'][i], axis=0) if seg_id in glove['timestamps'] else None,
        }
        data.append(row)
    return pd.DataFrame(data)