use super::support;

use anyhow::Result;
use bad_econ::simulation::research::capabilities::*;

#[test]
fn boundary_elimination_preserves_reachable_labels_across_cycles_and_empty_blocks() -> Result<()> {
    let device = support::device(64)?;
    let mut edges = vec![CapabilityEdge {
        prerequisite: 0,
        dependent: 2,
    }];
    edges.extend((2..39).map(|node| CapabilityEdge {
        prerequisite: node,
        dependent: node + 1,
    }));
    edges.extend([
        CapabilityEdge {
            prerequisite: 39,
            dependent: 1,
        },
        CapabilityEdge {
            prerequisite: 20,
            dependent: 10,
        },
    ]);
    let graph = CapabilityGraph::new(40, edges)?;
    let mut labels = vec![0; 40];
    labels[0] = 1;
    labels[1] = 4;
    labels[5] = 2;
    let packed = graph.upload(&device, &labels)?;
    let reduced = packed.eliminate(&device, 2)?;
    let observation = reduced.download(&device)?;
    assert_eq!(observation.labels, [1, 6]);
    assert_eq!(observation.adjacency, [2, 0]);
    let expected = graph.closure(&device, &labels)?.download(&device)?;
    assert_eq!(reduced.closure(&device)?.download(&device)?, expected[..2]);
    assert_eq!(
        packed
            .eliminate(&device, 40)?
            .closure(&device)?
            .download(&device)?,
        expected
    );
    assert!(
        packed
            .eliminate(&device, 0)?
            .closure(&device)?
            .download(&device)?
            .is_empty()
    );
    assert!(
        CapabilityGraph::new(0, vec![])?
            .upload(&device, &[])?
            .eliminate(&device, 0)?
            .download(&device)?
            .labels
            .is_empty()
    );
    assert!(packed.eliminate(&device, 41).is_err());
    Ok(())
}
