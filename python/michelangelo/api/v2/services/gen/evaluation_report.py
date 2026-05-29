from michelangelo.gen.api.v2.evaluation_report_svc_pb2_grpc import EvaluationReportServiceStub
from michelangelo.gen.api.v2.evaluation_report_svc_pb2 import (
    CreateEvaluationReportRequest,
    DeleteEvaluationReportCollectionRequest,
    DeleteEvaluationReportRequest,
    GetEvaluationReportRequest,
    ListEvaluationReportRequest,
    UpdateEvaluationReportRequest,
)
from michelangelo.gen.k8s.io.apimachinery.pkg.apis.meta.v1.generated_pb2 import (
    CreateOptions,
    DeleteOptions,
    GetOptions,
    ListOptions,
    UpdateOptions,
)

from ..base import BaseService, _TIMEOUT_SECONDS


class EvaluationReportService(BaseService):

    def __init__(self, context):
        super(EvaluationReportService, self).__init__(context, EvaluationReportServiceStub)

    def create_evaluation_report(self, evaluation_report, create_options=None, headers=None, timeout=_TIMEOUT_SECONDS):
        """
        Create an evaluation report.

        :param evaluation_report: evaluation report object
        :type evaluation_report: EvaluationReport
        :param create_options: create options
        :type create_options: Optional[Union[CreateOptions, Dict[str, Any]]]
        :param headers: request headers
        :type headers: Optional[Dict[str, str]]
        :param timeout: timeout in seconds, default is 60
        :type timeout: int

        :returns: created evaluation report
        :rtype: EvaluationReport

        :example:

        >>> from michelangelo.gen.api.v2.evaluation_report_pb2 import EvaluationReport
        >>> report = EvaluationReport()
        >>> report.metadata.namespace = 'my-project'
        >>> report.metadata.name = 'q1-eval'
        >>> APIClient.EvaluationReportService.create_evaluation_report(report)
        """
        req = CreateEvaluationReportRequest(evaluation_report=evaluation_report)
        create_options = self._process_message_or_dict(create_options, CreateOptions)
        req.create_options.CopyFrom(create_options)
        resp = self._stub.CreateEvaluationReport(req, metadata=self._get_metadata(headers), timeout=timeout)
        return resp.evaluation_report

    def get_evaluation_report(self, namespace, name, get_options=None, headers=None, timeout=_TIMEOUT_SECONDS):
        """
        Get an evaluation report by namespace and name.

        :param namespace: project name
        :type namespace: str
        :param name: evaluation report object name
        :type name: str
        :param get_options: get options
        :type get_options: Optional[Union[GetOptions, Dict[str, Any]]]
        :param headers: request headers
        :type headers: Optional[Dict[str, str]]
        :param timeout: timeout in seconds, default is 60
        :type timeout: int

        :returns: evaluation report
        :rtype: EvaluationReport

        :example:

        >>> report = APIClient.EvaluationReportService.get_evaluation_report(namespace='my-project', name='q1-eval')
        """
        req = GetEvaluationReportRequest(name=name, namespace=namespace)
        get_options = self._process_message_or_dict(get_options, GetOptions)
        req.get_options.CopyFrom(get_options)
        resp = self._stub.GetEvaluationReport(req, metadata=self._get_metadata(headers), timeout=timeout)
        return resp.evaluation_report

    def update_evaluation_report(self, evaluation_report, update_options=None, headers=None, timeout=_TIMEOUT_SECONDS):
        """
        Update an evaluation report.

        :param evaluation_report: evaluation report object
        :type evaluation_report: EvaluationReport
        :param update_options: update options
        :type update_options: Optional[Union[UpdateOptions, Dict[str, Any]]]
        :param headers: request headers
        :type headers: Optional[Dict[str, str]]
        :param timeout: timeout in seconds, default is 60
        :type timeout: int

        :returns: updated evaluation report
        :rtype: EvaluationReport

        :example:

        >>> report = APIClient.EvaluationReportService.get_evaluation_report(namespace='my-project', name='q1-eval')
        >>> report.spec.title = 'Updated Title'
        >>> APIClient.EvaluationReportService.update_evaluation_report(report)
        """
        req = UpdateEvaluationReportRequest(evaluation_report=evaluation_report)
        update_options = self._process_message_or_dict(update_options, UpdateOptions)
        req.update_options.CopyFrom(update_options)
        resp = self._stub.UpdateEvaluationReport(req, metadata=self._get_metadata(headers), timeout=timeout)
        return resp.evaluation_report

    def delete_evaluation_report(self, namespace, name, delete_options=None, headers=None, timeout=_TIMEOUT_SECONDS):
        """
        Delete an evaluation report.

        :param namespace: project name
        :type namespace: str
        :param name: evaluation report object name
        :type name: str
        :param delete_options: delete options
        :type delete_options: Optional[Union[DeleteOptions, Dict[str, Any]]]
        :param headers: request headers
        :type headers: Optional[Dict[str, str]]
        :param timeout: timeout in seconds, default is 60
        :type timeout: int

        :example:

        >>> APIClient.EvaluationReportService.delete_evaluation_report(namespace='my-project', name='q1-eval')
        """
        req = DeleteEvaluationReportRequest(namespace=namespace, name=name)
        delete_options = self._process_message_or_dict(delete_options, DeleteOptions)
        req.delete_options.CopyFrom(delete_options)
        self._stub.DeleteEvaluationReport(req, metadata=self._get_metadata(headers), timeout=timeout)

    def delete_evaluation_report_collection(self, namespace, delete_options=None, list_options=None, headers=None, timeout=_TIMEOUT_SECONDS):
        """
        Delete a collection of evaluation reports in a namespace.

        :param namespace: project name
        :type namespace: str
        :param delete_options: delete options
        :type delete_options: Optional[Union[DeleteOptions, Dict[str, Any]]]
        :param list_options: list options to filter which reports to delete
        :type list_options: Optional[Union[ListOptions, Dict[str, Any]]]
        :param headers: request headers
        :type headers: Optional[Dict[str, str]]
        :param timeout: timeout in seconds, default is 60
        :type timeout: int

        :example:

        >>> APIClient.EvaluationReportService.delete_evaluation_report_collection(namespace='my-project')
        """
        req = DeleteEvaluationReportCollectionRequest(namespace=namespace)
        delete_options = self._process_message_or_dict(delete_options, DeleteOptions)
        req.delete_options.CopyFrom(delete_options)
        list_options = self._process_message_or_dict(list_options, ListOptions)
        req.list_options.CopyFrom(list_options)
        self._stub.DeleteEvaluationReportCollection(req, metadata=self._get_metadata(headers), timeout=timeout)

    def list_evaluation_report(self, namespace, list_options=None, headers=None, timeout=_TIMEOUT_SECONDS):
        """
        List evaluation reports in a namespace.

        :param namespace: project name
        :type namespace: str
        :param list_options: list options
        :type list_options: Optional[Union[ListOptions, Dict[str, Any]]]
        :param headers: request headers
        :type headers: Optional[Dict[str, str]]
        :param timeout: timeout in seconds, default is 60
        :type timeout: int

        :returns: list of evaluation reports
        :rtype: EvaluationReportList

        :example:

        >>> reports = APIClient.EvaluationReportService.list_evaluation_report(namespace='my-project').items
        """
        req = ListEvaluationReportRequest(namespace=namespace)
        list_options = self._process_message_or_dict(list_options, ListOptions)
        req.list_options.CopyFrom(list_options)
        resp = self._stub.ListEvaluationReport(req, metadata=self._get_metadata(headers), timeout=timeout)
        return resp.evaluation_report_list
