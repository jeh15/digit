from pydrake.common.value import Value
from pydrake.multibody.plant import ContactResults
from pydrake.systems.framework import LeafSystem


class ContactReporter(LeafSystem):
    def __init__(self):
        super().__init__()  # Don't forget to initialize the base class.
        self.DeclareAbstractInputPort(
            name="contact_results",
            model_value=Value(
                ContactResults(),
                ),
            )
        # Calling `ForcedPublish()` will trigger the callback.
        self.DeclareForcedPublishEvent(self.Publish)

    def Publish(self, context):
        contact_results = self.get_input_port().Eval(context)

        num_point_contacts = contact_results.num_point_pair_contacts()
        num_hydroelastic_contacts = contact_results.num_hydroelastic_contacts()

        for c in range(num_hydroelastic_contacts):
            hydroelastic_contact_info = contact_results.hydroelastic_contact_info(c)
            spatial_force = hydroelastic_contact_info.F_Ac_W()
            contact_surface = hydroelastic_contact_info.contact_surface()
            num_faces = contact_surface.num_faces()
            total_area = contact_surface.total_area()
            centroid = contact_surface.centroid()


def add_contact_report(builder, plant):
    contact_reporter = builder.AddSystem(ContactReporter())
    builder.Connect(
        plant.get_contact_results_output_port(),
        contact_reporter.get_input_port(0),
    )

    return builder, plant
