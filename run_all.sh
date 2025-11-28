poetry install
poetry run setup
cp ./functions.csv ./SmartEmbed/functions.csv

# smartembed setup
cd ./SmartEmbed
mkdir output
docker build --no-cache -t smartembed .
docker run --rm -it  -v $(pwd)/output:/app/output  smartembed:latest
cd ../
cp ./SmartEmbed/output/SmartEmbed_embeddings.csv ./data/SmartEmbed_embeddings.csv

#nicad setup
cp -r queries ./solidity-nicad/
cp -r documents ./solidity-nicad/
cd ./solidity-nicad
docker/run_cross_clones
cd ../
cp ./solidity-nicad/queries_functions-consistent-crossclones/queries_functions-consistent-crossclones-0.30.xml ./nicad_results.xml

#train
poetry run train

#eval
poetry run eval