"""
 Due to bandwidth limitations, the web interface (https://neurosynth.org/) 
 is not intended to support mass downloading of hundreds or thousands of 
 images, and attempts to scrape content in an automated way will result 
 in permanent IP bans.
 Source: check Data from https://neurosynth.org/code/

 Therefore we download manually each html showing the max number (100) of 
 title under the tag 'studies' for each topic inside 
 https://neurosynth.org/analyses/topics/v5-topics-200/ for example.
 Donwload all the files to out_dir/v5topic200
 Download the root pages of the topics 
 (https://neurosynth.org/analyses/topics/v5-topics-200/) as:
    - Neurosynth_v5topic200_00_topics.htm
    - Neurosynth_v5topic200_01_topics.htm
    - ...
 Download the pages of each topic 
 (https://neurosynth.org/analyses/topics/v5-topics-200/0) as:
    - Neurosynth_v5topic200_topic000_00.htm
    - Neurosynth_v5topic200_topic000_01.htm
    - ...
"""
import glob
import os
import nimare
import numpy as np
from bs4 import BeautifulSoup
from neurosynth.base.dataset import download

# Download Neurosynth
out_dir = os.path.abspath('/Users/jperaza/Desktop/neurosynth/')

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
if not os.path.isfile(os.path.join(out_dir, 'database.txt')):
    download(out_dir, unpack=True)

# Convert Neurosynth database files to NiMARE Dataset
dset = nimare.io.convert_neurosynth_to_dataset(
    os.path.join(out_dir, 'database.txt'),
    os.path.join(out_dir, 'features.txt'))

dset.save(os.path.join(out_dir, 'neurosynth_dataset.pkl.gz'))

TOPICS = 'v5topic200'
for topic in range(200):
    files = glob.glob(os.path.join(
        out_dir, TOPICS, 'Neurosynth_v5topic200_topic{:03d}_*.htm'.format(topic)))
    files.sort()
    n_topics = glob.glob(os.path.join(
        out_dir, TOPICS, 'Neurosynth_v5topic200_*_topics.htm'))
    n_topics.sort()

    # Find the number of studies by topic to perform a final check
    n_studies = []
    for n_topic in n_topics:
        with open(n_topic) as html_topics:
            soup_topics = BeautifulSoup(html_topics, 'lxml')
        studies_table = soup_topics.find('div', class_='row').find(
            'div', class_='col-md-12 content').find_all('td')

        [n_studies.append(int(studies_table[idx*3+2].text))
         for idx in range(int(len(studies_table)/3))]

    nimare_ids_weight = {}
    for file in files:
        with open(file) as html_topic:
            soup_topic = BeautifulSoup(html_topic, 'lxml')

        studies_table = soup_topic.find('div', class_='row').find(
            'div', class_='row').find('div', class_='col-md-10 content').find(
                'div', class_='tab-content').find('div', id='studies').table.tbody
        studies = studies_table.find_all('a')
        weights = studies_table.find_all('td', class_='sorting_1')

        for i, study in enumerate(studies):
            expid = '1'
            pid = study['href'].split('/')[4]
            nimare_ids_weight["{0}-{1}".format(pid, expid)] = weights[i].text

    found_ids = np.isin(dset.ids, list(nimare_ids_weight.keys()))
    ids_colum = np.zeros(len(dset.ids))
    topic_weights = [nimare_ids_weight[id] for id in dset.ids[found_ids]]
    ids_colum[found_ids] = topic_weights
    
    # Doble check that all the studies are in the dataset and match the number of
    # studies reported on https://neurosynth.org/analyses/topics/v5-topics-200/
    nonzero = ids_colum[ids_colum != 0]
    if len(nonzero) != n_studies[topic]:
        print('Only {} out of {} studies found in topic {} from {}'.format(
            len(nonzero), n_studies[topic], topic, TOPICS))
        print('Check local html file')

    # Add annotation to Dataset and save to file
    dset.annotations['Neurosynth_{}__topic{:03d}'.format(
        TOPICS, topic)] = ids_colum

dset.save(os.path.join(out_dir, 'neurosynth_dataset_annotation_test.pkl.gz'))
