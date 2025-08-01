mkdir source_dataset
wget --header="User-Agent: Mozilla/5.0" "https://figshare.com/ndownloader/articles/20780878/versions/2" -O ./source_dataset/dataset.zip
cd source_dataset
unzip dataset
unzip verified-smart-contract-code-comments.zip
rm dataset
rm verified-smart-contract-code-comments.zip
cd ../
mkdir data
wget --header="User-Agent: Mozilla/5.0" "https://figshare.com/ndownloader/articles/29617757?private_link=46e03fcb85b6ec2ba08c" -O ./data/decompiled_dataset.zip
cd data
unzip decompiled_dataset.zip
unzip bytecode_dataset.zip
unzip decomp_datasets.zip
rm decompiled_dataset.zip
rm bytecode_dataset.zip
rm decomp_datasets.zip
