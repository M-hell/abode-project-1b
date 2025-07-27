docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .

do changes in input 

docker run --rm -v ${PWD}/input:/app/input -v ${PWD}/output:/app/output --network none mysolutionname:somerandomidentifier