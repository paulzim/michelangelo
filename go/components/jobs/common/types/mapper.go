package types

import (
	"k8s.io/apimachinery/pkg/runtime"

	v2pb "github.com/michelangelo-ai/michelangelo/proto-go/api/v2"
)

// Mapper abstracts mapping global job objects (Michelangelo representations)
// to their local (Kubernetes) counterparts and back. Implementations live
// alongside specific compute backends (e.g. k8sengine for the in-cluster
// KubeRay path); consumers depend on this interface, never on a concrete impl.
type Mapper interface {
	// MapGlobalJobToLocal converts a global job object and its associated cluster
	// object into a Kubernetes-native runtime.Object representing the job.
	MapGlobalJobToLocal(jobObject runtime.Object, jobClusterObject runtime.Object, cluster *v2pb.Cluster) (runtime.Object, error)

	// MapGlobalJobClusterToLocal converts a global cluster object into a
	// Kubernetes-native runtime.Object representing the cluster.
	MapGlobalJobClusterToLocal(jobClusterObject runtime.Object, cluster *v2pb.Cluster) (runtime.Object, error)

	// GetLocalName extracts the namespace and name from the provided job object.
	GetLocalName(obj runtime.Object) (namespace, name string)

	// MapLocalClusterStatusToGlobal converts a local (Kubernetes) cluster status
	// object to the global Michelangelo ClusterStatus representation.
	MapLocalClusterStatusToGlobal(localClusterObject runtime.Object) (*JobClusterStatus, error)

	// MapLocalJobStatusToGlobal converts a local (Kubernetes) job status object
	// to the global Michelangelo JobStatus representation.
	MapLocalJobStatusToGlobal(localJobObject runtime.Object) (*JobStatus, error)
}
