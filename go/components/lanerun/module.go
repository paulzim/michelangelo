package lanerun

import (
	"go.uber.org/config"
	"go.uber.org/fx"
	"go.uber.org/zap"
	"sigs.k8s.io/controller-runtime/pkg/manager"

	apiHandler "github.com/michelangelo-ai/michelangelo/go/api/handler"
	maconfig "github.com/michelangelo-ai/michelangelo/go/base/config"
)

const configKey = "lanerun"

// Module is the Uber FX module for the LaneRun controller.
//
// It depends on maconfig.InferenceServerConfig already being provided in the
// fx graph (via inferenceserver's endpoints.Module) — the LaneRun controller
// calls the Pit Crew Advisor's InferenceServer through the same gateway the
// InferenceServer controller resolves, so it deliberately reuses that config
// rather than defining its own. Include this module alongside
// inferenceserver.Module:
//
//	fx.New(
//	    inferenceserver.Module,
//	    lanerun.Module,
//	    // other modules...
//	)
var Module = fx.Options(
	fx.Provide(newConfig),
	fx.Invoke(register),
)

func newConfig(provider config.Provider) (Config, error) {
	cfg := Config{}
	err := provider.Get(configKey).Populate(&cfg)
	return cfg, err
}

type registerParams struct {
	fx.In

	Mgr               manager.Manager
	APIHandlerFactory apiHandler.Factory
	Logger            *zap.Logger
	GatewayConfig     maconfig.InferenceServerConfig
	Config            Config
}

func register(p registerParams) error {
	return NewReconciler(p.APIHandlerFactory, p.Logger, p.GatewayConfig, p.Config).Register(p.Mgr)
}
