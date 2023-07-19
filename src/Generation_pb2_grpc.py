# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import Generation_pb2 as Generation__pb2


class image_generationStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Gen = channel.unary_unary(
                '/image_generation.image_generation/Gen',
                request_serializer=Generation__pb2.Text.SerializeToString,
                response_deserializer=Generation__pb2.Image.FromString,
                )


class image_generationServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Gen(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_image_generationServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Gen': grpc.unary_unary_rpc_method_handler(
                    servicer.Gen,
                    request_deserializer=Generation__pb2.Text.FromString,
                    response_serializer=Generation__pb2.Image.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'image_generation.image_generation', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class image_generation(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Gen(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/image_generation.image_generation/Gen',
            Generation__pb2.Text.SerializeToString,
            Generation__pb2.Image.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
