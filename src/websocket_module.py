from dataclasses import dataclass
import json
import time

from ws4py.client.threadedclient import WebSocketClient

from pydrake.common.value import AbstractValue
from pydrake.systems.framework import (
    LeafSystem,
    Context,
)


@dataclass
class MessageWrapper:
    message: str


def make_message_wrapper_value(message: str):
    return AbstractValue.Make(MessageWrapper(message))


class BasicClient(WebSocketClient):
    def opened(self):
        self.operation_mode = None
        self.responded = True

        privilege_request = [
            "request-privilege",
            {"privilege": "change-action-command", "priority": 0},
        ]
        self.send(json.dumps(privilege_request))

        self.llapi_enabled = False

    def closed(self, code, reason):
        print(("Closed", code, reason))

    def received_message(self, m):
        dataloaded = json.loads(m.data)
        message_type = str(dataloaded[0])
        message_dict = dataloaded[1]

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
                self.llapi_enabled = True

                print(
                    (
                        "action-status-changed to low-level-api: ",
                        self.status_change_mode,
                    )
                )
                print("Transition successful")
            if self.status_change_mode == "failure":
                self.llapi_enabled = False

                print(
                    (
                        "action-status-changed to low-level-api: ",
                        self.status_change_mode,
                    )
                )
                print(
                    "Cannot transition to low-level-api operation mode if no low-level commands are present"
                )
                time.sleep(1)


request_robot_status = ["get-robot-status", {}, 3]

privilege_request = [
    "request-privilege",
    {"privilege": "change-action-command", "priority": 0},
]

request_lowlevelapi = [
    "action-set-operation-mode",
    {"mode": "low-level-api"},
    1,
]

request_lowlevelapi_off = [
    "action-set-operation-mode",
    {"mode": "locomotion"},
    2,
]

request_shutoff = [
    "action-set-operation-mode",
    {"mode": "damping"},
]


class WebsocketModule(LeafSystem):
    """
        WebsocketModule that recieves messages from other LeafSystems
        to interact with Agility's JSON API Messaging sytem.

        Methods:
            initialize low level api and send out message when it has been initialized.
            switchs locomotion mode when safety controller reports and error.

    """
    def __init__(self, ip_address: str, port: int):
        super().__init__()
        socket_address = f'ws://{ip_address}:{port}/'
        self.ws = BasicClient(socket_address, protocols=["json-v1-agility"])
        self.message = ''

        # Input Port: Message
        self.message_port = self.DeclareAbstractInputPort(
            "messsage",
            make_message_wrapper_value(self.message),
        ).get_index()

        # self.operation_port = self.DeclareAbstractOutputPort(
        #     "plant_context",
        #     alloc=lambda: make_context_wrapper_value(plant),
        #     calc=calc_context,
        # ).get_index()

        def on_periodic(context, event):
            self.message = self.get_input_port(
                self.message_port,
            ).Eval(context).message

            if (
                self.message == 'low-level-api' and
                self.ws.operation_mode != 'low-level-api'
            ):
                ws.send(json.dumps(request_lowlevelapi))

            if self.message == 'shutoff':
                ws.send(json.dumps(request_shutoff))


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
                    input_port
                    .Eval(context)
                    .message
                )

            # Handle Message:
            if 'shutoff' in messages:
                output.set_mutable_value(
                    make_message_wrapper_value('shutoff')
                )
            elif 'low-level-api' in messages:
                output.set_mutable_value(
                    make_message_wrapper_value('low-level-api')
                )
            else:
                output.set_mutable_value(
                    make_message_wrapper_value('')
                )

        self.message_port = self.DeclareAbstractOutputPort(
            "message",
            alloc=lambda: make_message_wrapper_value(message),
            calc=handle_messages,
        ).get_index()





if __name__ == "__main__":
    ws = BasicClient("ws://localhost:8080/", protocols=["json-v1-agility"])
    ws.connect()

    ws.send(json.dumps(request_robot_status))
    ws.send(json.dumps(request_shutoff))
