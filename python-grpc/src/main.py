import logging
import time
import grpc
from concurrent import futures
from protos.service_pb2_grpc import ClusteringServiceServicer, add_ClusteringServiceServicer_to_server
from protos.service_pb2 import PredictResponse


class ClusteringService(ClusteringServiceServicer):

    def Predict(self, request, context):
        print("Hey!!")
        return PredictResponse(cluster_index=1)


def serve():
    # Initialize GRPC Server
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # Initialize Services
    add_ClusteringServiceServicer_to_server(ClusteringService(), grpc_server)
    # Start GRPC Server
    grpc_server.add_insecure_port('[::]:5001')
    grpc_server.start()
    # Keep application alive
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        grpc_server.stop(0)


if __name__ == "__main__":
    logging.basicConfig()
    serve()
