package lanerun

import "time"

// Config holds runtime configuration for the LaneRun controller.
type Config struct {
	// AdvisorNamespace is the namespace containing the Pit Crew Advisor's
	// InferenceServer and Deployment.
	AdvisorNamespace string `yaml:"advisorNamespace"`

	// AdvisorInferenceServerName is the `metadata.name` of the InferenceServer
	// serving the Pit Crew Advisor model.
	AdvisorInferenceServerName string `yaml:"advisorInferenceServerName"`

	// AdvisorDeploymentName is the `metadata.name` of the Deployment serving
	// the Pit Crew Advisor model on that InferenceServer.
	AdvisorDeploymentName string `yaml:"advisorDeploymentName"`

	// AdvisorTimeout bounds how long the controller waits for the advisor's
	// /infer response before treating the query as failed. Defaults to 5s
	// when zero.
	AdvisorTimeout time.Duration `yaml:"advisorTimeout"`
}
