# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import sys
import os
# 将主目录添加到 sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import eventInfo.event_pb2 as event__pb2


class EventServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.FaceRecognitionEvent = channel.unary_unary(
                '/event.EventService/FaceRecognitionEvent',
                request_serializer=event__pb2.FaceRecognitionEventRequest.SerializeToString,
                response_deserializer=event__pb2.EventServerResopnse.FromString,
                )
        self.EmotionDetectionEvent = channel.unary_unary(
                '/event.EventService/EmotionDetectionEvent',
                request_serializer=event__pb2.EmotionDetectionEventRequest.SerializeToString,
                response_deserializer=event__pb2.EventServerResopnse.FromString,
                )
        self.VolunteerInteractionEvent = channel.unary_unary(
                '/event.EventService/VolunteerInteractionEvent',
                request_serializer=event__pb2.VolunteerInteractionEventRequest.SerializeToString,
                response_deserializer=event__pb2.EventServerResopnse.FromString,
                )
        self.StrangerDetectionEvent = channel.unary_unary(
                '/event.EventService/StrangerDetectionEvent',
                request_serializer=event__pb2.StrangerDetectionEventRequest.SerializeToString,
                response_deserializer=event__pb2.EventServerResopnse.FromString,
                )
        self.FallDetectionEvent = channel.unary_unary(
                '/event.EventService/FallDetectionEvent',
                request_serializer=event__pb2.FallDetectionEventRequest.SerializeToString,
                response_deserializer=event__pb2.EventServerResopnse.FromString,
                )
        self.ForbiddenAreaInvasionDetectionEvent = channel.unary_unary(
                '/event.EventService/ForbiddenAreaInvasionDetectionEvent',
                request_serializer=event__pb2.ForbiddenAreaInvasionDetectionEventRequest.SerializeToString,
                response_deserializer=event__pb2.EventServerResopnse.FromString,
                )
        self.FireDetectionEvent = channel.unary_unary(
                '/event.EventService/FireDetectionEvent',
                request_serializer=event__pb2.FireDetectionEventRequest.SerializeToString,
                response_deserializer=event__pb2.EventServerResopnse.FromString,
                )
        self.UpdateImageUrl = channel.unary_unary(
                '/event.EventService/UpdateImageUrl',
                request_serializer=event__pb2.UpdateImageUrlRequest.SerializeToString,
                response_deserializer=event__pb2.EventServerResopnse.FromString,
                )


class EventServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def FaceRecognitionEvent(self, request, context):
        """rpc CreateEvent(EventRequest) returns (EventServerResopnse);
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def EmotionDetectionEvent(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def VolunteerInteractionEvent(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StrangerDetectionEvent(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def FallDetectionEvent(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ForbiddenAreaInvasionDetectionEvent(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def FireDetectionEvent(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateImageUrl(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_EventServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'FaceRecognitionEvent': grpc.unary_unary_rpc_method_handler(
                    servicer.FaceRecognitionEvent,
                    request_deserializer=event__pb2.FaceRecognitionEventRequest.FromString,
                    response_serializer=event__pb2.EventServerResopnse.SerializeToString,
            ),
            'EmotionDetectionEvent': grpc.unary_unary_rpc_method_handler(
                    servicer.EmotionDetectionEvent,
                    request_deserializer=event__pb2.EmotionDetectionEventRequest.FromString,
                    response_serializer=event__pb2.EventServerResopnse.SerializeToString,
            ),
            'VolunteerInteractionEvent': grpc.unary_unary_rpc_method_handler(
                    servicer.VolunteerInteractionEvent,
                    request_deserializer=event__pb2.VolunteerInteractionEventRequest.FromString,
                    response_serializer=event__pb2.EventServerResopnse.SerializeToString,
            ),
            'StrangerDetectionEvent': grpc.unary_unary_rpc_method_handler(
                    servicer.StrangerDetectionEvent,
                    request_deserializer=event__pb2.StrangerDetectionEventRequest.FromString,
                    response_serializer=event__pb2.EventServerResopnse.SerializeToString,
            ),
            'FallDetectionEvent': grpc.unary_unary_rpc_method_handler(
                    servicer.FallDetectionEvent,
                    request_deserializer=event__pb2.FallDetectionEventRequest.FromString,
                    response_serializer=event__pb2.EventServerResopnse.SerializeToString,
            ),
            'ForbiddenAreaInvasionDetectionEvent': grpc.unary_unary_rpc_method_handler(
                    servicer.ForbiddenAreaInvasionDetectionEvent,
                    request_deserializer=event__pb2.ForbiddenAreaInvasionDetectionEventRequest.FromString,
                    response_serializer=event__pb2.EventServerResopnse.SerializeToString,
            ),
            'FireDetectionEvent': grpc.unary_unary_rpc_method_handler(
                    servicer.FireDetectionEvent,
                    request_deserializer=event__pb2.FireDetectionEventRequest.FromString,
                    response_serializer=event__pb2.EventServerResopnse.SerializeToString,
            ),
            'UpdateImageUrl': grpc.unary_unary_rpc_method_handler(
                    servicer.UpdateImageUrl,
                    request_deserializer=event__pb2.UpdateImageUrlRequest.FromString,
                    response_serializer=event__pb2.EventServerResopnse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'event.EventService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class EventService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def FaceRecognitionEvent(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/event.EventService/FaceRecognitionEvent',
            event__pb2.FaceRecognitionEventRequest.SerializeToString,
            event__pb2.EventServerResopnse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def EmotionDetectionEvent(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/event.EventService/EmotionDetectionEvent',
            event__pb2.EmotionDetectionEventRequest.SerializeToString,
            event__pb2.EventServerResopnse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def VolunteerInteractionEvent(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/event.EventService/VolunteerInteractionEvent',
            event__pb2.VolunteerInteractionEventRequest.SerializeToString,
            event__pb2.EventServerResopnse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StrangerDetectionEvent(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/event.EventService/StrangerDetectionEvent',
            event__pb2.StrangerDetectionEventRequest.SerializeToString,
            event__pb2.EventServerResopnse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def FallDetectionEvent(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/event.EventService/FallDetectionEvent',
            event__pb2.FallDetectionEventRequest.SerializeToString,
            event__pb2.EventServerResopnse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ForbiddenAreaInvasionDetectionEvent(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/event.EventService/ForbiddenAreaInvasionDetectionEvent',
            event__pb2.ForbiddenAreaInvasionDetectionEventRequest.SerializeToString,
            event__pb2.EventServerResopnse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def FireDetectionEvent(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/event.EventService/FireDetectionEvent',
            event__pb2.FireDetectionEventRequest.SerializeToString,
            event__pb2.EventServerResopnse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateImageUrl(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/event.EventService/UpdateImageUrl',
            event__pb2.UpdateImageUrlRequest.SerializeToString,
            event__pb2.EventServerResopnse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
