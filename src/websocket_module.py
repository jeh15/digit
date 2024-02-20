from dataclasses import dataclass
import json

from ws4py.client.threadedclient import WebSocketClient

from pydrake.common.value import AbstractValue
from pydrake.systems.framework import (
    LeafSystem,
    PublishEvent,
    TriggerType,
)


@dataclass
class MessageWrapper:
    message: str


def make_message_wrapper_value(message: str):
    return AbstractValue.Make(MessageWrapper(message))


class BasicClient(WebSocketClient):
    operation_mode = None

    def opened(self):
        self.operation_mode = None
        self.responded = True

        privilege_request = [
            "request-privilege",
            {"privilege": "change-action-command", "priority": 0},
        ]
        self.send(json.dumps(privilege_request))

    def closed(self, code, reason):
        print(("Closed", code, reason))

    def received_message(self, m):
        dataloaded = json.loads(m.data)
        message_type = str(dataloaded[0])
        message_dict = dataloaded[1]

        self.data = dataloaded

        if message_type == "privileges":
            self.done = message_dict["privileges"][0]["has"]
            if self.done:
                print(("Privilege request executed successfully!"))

        if message_type == "robot-status":
            self.responded = True
            self.operation_mode = str(message_dict["operation-mode"])
            self.left_foot_contact = message_dict["left-foot-in-contact"]
            self.right_foot_contact = message_dict["right-foot-in-contact"]

        if message_type == "error":
            self.error_info = str(message_dict["info"])
            print(("Error: ", self.error_info))

        if message_type == "action-status-changed":
            self.status_change_mode = str(message_dict["status"])

            if self.status_change_mode == "success":
                print(
                    (
                        "action-status-changed",
                        self.status_change_mode,
                    )
                )
                print("Transition successful")
            if self.status_change_mode == "failure":
                print(
                    (
                        "action-status-changed: ",
                        self.status_change_mode,
                    )
                )


class WebsocketModule(LeafSystem):
    """
        WebsocketModule that recieves messages from other LeafSystems
        to interact with Agility's JSON API Messaging sytem.

        Methods:
            initialize low level api and send out message when it has been initialized.
            switchs locomotion mode when safety controller reports an error.

    """
    def __init__(self, ip_address: str, port: int, update_rate: float = 1e-4):
        super().__init__()
        socket_address = f'ws://{ip_address}:{port}/'
        self.ws = BasicClient(socket_address, protocols=["json-v1-agility"])
        self.message = ''
        self.update_rate = update_rate

        # Message to requests:
        self.request_robot_status = ["get-robot-status", {}, 3]
        self.privilege_request = [
            "request-privilege",
            {"privilege": "change-action-command", "priority": 0},
        ]
        self.request_lowlevelapi = [
            "action-set-operation-mode",
            {"mode": "low-level-api"},
            1,
        ]
        self.request_lowlevelapi_off = [
            "action-set-operation-mode",
            {"mode": "locomotion"},
            2,
        ]
        self.request_shutdown = [
            "action-set-operation-mode",
            {"mode": "damping"},
        ]

        # Input Port: Message
        self.message_port = self.DeclareAbstractInputPort(
            "messsage",
            make_message_wrapper_value(self.message),
        ).get_index()

        def on_initialization(context, event):
            self.ws.connect()
            # Initialize Robot Status:
            self.ws.send(json.dumps(self.request_robot_status))

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialization,
            ),
        )

        def on_periodic(context, event):
            self.message = (
                self.get_input_port(
                    self.message_port,
                )
                .Eval(context)
                .message
            )

            # If not in damping mode, send message to websocket server:
            if self.ws.operation_mode != 'damping':
                # If message is not empty, send message to websocket server:
                if self.message:
                    if (
                        self.message == 'low-level-api' and
                        self.ws.operation_mode != 'low-level-api'
                    ):
                        self.ws.send(json.dumps(self.request_lowlevelapi))

                    if self.message == 'shutdown':
                        self.ws.send(json.dumps(self.request_shutdown))

            # Send robot status request:
            self.ws.send(json.dumps(self.request_robot_status))

        self.DeclarePeriodicEvent(
            period_sec=self.update_rate,
            offset_sec=self.update_rate,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=on_periodic,
            ),
        )

    def __del__(self):
        self.ws.close()


class MessageHandler(LeafSystem):
    def __init__(self, num_messengers: int):
        super().__init__()
        self.input_ports = []
        message = ''
        for i in range(num_messengers):
            name = f'input_port_{i}'
            self.input_ports.append(
                self.DeclareAbstractInputPort(
                    name,
                    make_message_wrapper_value(message),
                ).get_index()
            )

        def handle_messages(context, output):
            messages = []
            for input_port in self.input_ports:
                messages.append(
                    self.get_input_port(input_port)
                    .Eval(context)
                    .message
                )

            # Handle Message:
            if 'shutdown' in messages:
                output.set_value(
                    MessageWrapper('shutdown'),
                )
            elif 'low-level-api' in messages:
                output.set_value(
                    MessageWrapper('low-level-api'),
                )
            else:
                output.set_value(
                    MessageWrapper(''),
                )

        self.message_port = self.DeclareAbstractOutputPort(
            "message",
            alloc=lambda: make_message_wrapper_value(message),
            calc=handle_messages,
        ).get_index()


class MessagePublisher(LeafSystem):
    def __init__(self):
        super().__init__()
        self.message = ''

        def publish_message(context, output):
            output.set_value(
                MessageWrapper(self.message),
            )

        self.message_port = self.DeclareAbstractOutputPort(
            "message",
            alloc=lambda: make_message_wrapper_value(self.message),
            calc=publish_message,
        ).get_index()


if __name__ == "__main__":
    ws = BasicClient("ws://localhost:8080/", protocols=["json-v1-agility"])
    ws.connect()

    ws.send(json.dumps(request_robot_status))
    ws.send(json.dumps(request_shutdown))
