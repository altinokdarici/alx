use alx_protocol::Event;

/// Where the core sends [`Event`]s. This is the seam every UI plugs into.
pub trait EventSink: Send + Sync {
    fn emit(&self, event: Event);
}

/// Forward events over a channel, e.g. to a server task streaming them to
/// remote clients.
impl EventSink for tokio::sync::mpsc::UnboundedSender<Event> {
    fn emit(&self, event: Event) {
        // A closed receiver means nobody is listening, which is not an error
        // for the agent.
        let _ = self.send(event);
    }
}

/// Discards all events.
pub struct NullSink;

impl EventSink for NullSink {
    fn emit(&self, _event: Event) {}
}
