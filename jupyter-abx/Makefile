TAG ?= $(shell git describe --tags --always --dirty)

docker-build-jupyter-scipy:
	make docker-build -C ../jupyter-scipy TAG=${TAG} 

docker-build: docker-build-jupyter-scipy
	docker build -t jupyter-abx:${TAG} --build-arg BASE_IMG=jupyter-scipy:${TAG} -f Dockerfile . 

docker-push:
	docker push jupyter-abx:${TAG}