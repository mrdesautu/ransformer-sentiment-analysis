#!/usr/bin/env python3
"""
🧪 Test Suite para la Interfaz Web del Transformer
Pruebas automatizadas para verificar funcionalidad y rendimiento
"""

import requests
import json
import time
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

class WebInterfaceTestSuite:
    def __init__(self, api_url: str = "http://127.0.0.1:8000", web_url: str = "http://localhost:8080"):
        self.api_url = api_url
        self.web_url = web_url
        self.session = requests.Session()
        self.test_results = []
        
    def log_test(self, test_name: str, passed: bool, message: str = "", duration: float = 0):
        """Registra resultado de un test"""
        status = "✅ PASS" if passed else "❌ FAIL"
        result = {
            "test": test_name,
            "passed": passed,
            "message": message,
            "duration": duration
        }
        self.test_results.append(result)
        print(f"{status} {test_name} ({duration:.2f}s) - {message}")
        
    def test_api_health(self) -> bool:
        """Test: API Health Check"""
        start_time = time.time()
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log_test("API Health Check", True, "API respondiendo correctamente", duration)
                    return True
                else:
                    self.log_test("API Health Check", False, "Estado de salud inválido", duration)
                    return False
            else:
                self.log_test("API Health Check", False, f"Status code: {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("API Health Check", False, f"Error: {str(e)}", duration)
            return False
    
    def test_web_interface_loading(self) -> bool:
        """Test: Carga de la interfaz web"""
        start_time = time.time()
        try:
            response = self.session.get(self.web_url, timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                if "Transformer" in response.text and "sentiment" in response.text.lower():
                    self.log_test("Web Interface Loading", True, "Interfaz cargada correctamente", duration)
                    return True
                else:
                    self.log_test("Web Interface Loading", False, "Contenido incorrecto", duration)
                    return False
            else:
                self.log_test("Web Interface Loading", False, f"Status code: {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Web Interface Loading", False, f"Error: {str(e)}", duration)
            return False
    
    def test_single_prediction(self) -> bool:
        """Test: Predicción individual"""
        start_time = time.time()
        test_text = "I love this amazing product!"
        
        try:
            payload = {"text": test_text}
            response = self.session.post(f"{self.api_url}/predict", json=payload, timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if "sentiment" in data and "confidence" in data:
                    sentiment = data["sentiment"]
                    confidence = data["confidence"]
                    if sentiment in ["POSITIVE", "NEGATIVE"] and 0 <= confidence <= 1:
                        self.log_test("Single Prediction", True, f"Sentiment: {sentiment}, Confidence: {confidence:.3f}", duration)
                        return True
                    else:
                        self.log_test("Single Prediction", False, "Formato de respuesta inválido", duration)
                        return False
                else:
                    self.log_test("Single Prediction", False, "Campos faltantes en respuesta", duration)
                    return False
            else:
                self.log_test("Single Prediction", False, f"Status code: {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Single Prediction", False, f"Error: {str(e)}", duration)
            return False
    
    def test_batch_prediction(self) -> bool:
        """Test: Predicción por lotes"""
        start_time = time.time()
        test_texts = [
            "This is amazing!",
            "I hate this product.",
            "It's okay, nothing special."
        ]
        
        try:
            payload = {"texts": test_texts}
            response = self.session.post(f"{self.api_url}/predict/batch", json=payload, timeout=15)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if "predictions" in data and len(data["predictions"]) == len(test_texts):
                    predictions = data["predictions"]
                    valid_predictions = all(
                        "sentiment" in pred and "confidence" in pred 
                        for pred in predictions
                    )
                    if valid_predictions:
                        self.log_test("Batch Prediction", True, f"Procesados {len(predictions)} textos", duration)
                        return True
                    else:
                        self.log_test("Batch Prediction", False, "Predicciones inválidas", duration)
                        return False
                else:
                    self.log_test("Batch Prediction", False, "Formato de respuesta incorrecto", duration)
                    return False
            else:
                self.log_test("Batch Prediction", False, f"Status code: {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Batch Prediction", False, f"Error: {str(e)}", duration)
            return False
    
    def test_probabilities_endpoint(self) -> bool:
        """Test: Endpoint de probabilidades"""
        start_time = time.time()
        test_text = "This movie is fantastic!"
        
        try:
            payload = {"text": test_text}
            response = self.session.post(f"{self.api_url}/predict/probabilities", json=payload, timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if "probabilities" in data:
                    probs = data["probabilities"]
                    if "POSITIVE" in probs and "NEGATIVE" in probs:
                        total_prob = probs["POSITIVE"] + probs["NEGATIVE"]
                        if abs(total_prob - 1.0) < 0.01:  # Tolerancia de flotantes
                            self.log_test("Probabilities Endpoint", True, f"Probs: {probs}", duration)
                            return True
                        else:
                            self.log_test("Probabilities Endpoint", False, f"Probabilidades no suman 1: {total_prob}", duration)
                            return False
                    else:
                        self.log_test("Probabilities Endpoint", False, "Clases de probabilidad faltantes", duration)
                        return False
                else:
                    self.log_test("Probabilities Endpoint", False, "Campo 'probabilities' faltante", duration)
                    return False
            else:
                self.log_test("Probabilities Endpoint", False, f"Status code: {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Probabilities Endpoint", False, f"Error: {str(e)}", duration)
            return False
    
    def test_model_info(self) -> bool:
        """Test: Información del modelo"""
        start_time = time.time()
        try:
            response = self.session.get(f"{self.api_url}/model/info", timeout=5)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["model_name", "model_type", "num_parameters"]
                if all(field in data for field in required_fields):
                    self.log_test("Model Info", True, f"Modelo: {data.get('model_name')}", duration)
                    return True
                else:
                    self.log_test("Model Info", False, "Campos requeridos faltantes", duration)
                    return False
            else:
                self.log_test("Model Info", False, f"Status code: {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Model Info", False, f"Error: {str(e)}", duration)
            return False
    
    def test_web_static_files(self) -> bool:
        """Test: Archivos estáticos de la web"""
        start_time = time.time()
        static_files = [
            "/styles.css",
            "/app.js",
            "/config.json"
        ]
        
        failed_files = []
        for file_path in static_files:
            try:
                response = self.session.get(f"{self.web_url}{file_path}", timeout=5)
                if response.status_code != 200:
                    failed_files.append(file_path)
            except Exception:
                failed_files.append(file_path)
        
        duration = time.time() - start_time
        
        if not failed_files:
            self.log_test("Web Static Files", True, f"Todos los archivos cargados ({len(static_files)})", duration)
            return True
        else:
            self.log_test("Web Static Files", False, f"Archivos fallidos: {failed_files}", duration)
            return False
    
    def test_performance_load(self, num_requests: int = 10) -> bool:
        """Test: Rendimiento bajo carga"""
        start_time = time.time()
        test_text = "Performance test text"
        
        def make_request():
            try:
                payload = {"text": test_text}
                response = self.session.post(f"{self.api_url}/predict", json=payload, timeout=10)
                return response.status_code == 200
            except Exception:
                return False
        
        try:
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request) for _ in range(num_requests)]
                results = [future.result() for future in as_completed(futures)]
            
            duration = time.time() - start_time
            success_rate = sum(results) / len(results)
            avg_response_time = duration / num_requests
            
            if success_rate >= 0.9:  # 90% de éxito
                self.log_test("Performance Load", True, f"Success rate: {success_rate:.1%}, Avg time: {avg_response_time:.3f}s", duration)
                return True
            else:
                self.log_test("Performance Load", False, f"Success rate: {success_rate:.1%} (< 90%)", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Performance Load", False, f"Error: {str(e)}", duration)
            return False
    
    def test_error_handling(self) -> bool:
        """Test: Manejo de errores"""
        start_time = time.time()
        
        # Test con texto vacío
        try:
            payload = {"text": ""}
            response = self.session.post(f"{self.api_url}/predict", json=payload, timeout=5)
            empty_text_handled = response.status_code in [400, 422]
        except Exception:
            empty_text_handled = False
        
        # Test con texto muy largo
        try:
            payload = {"text": "a" * 10000}
            response = self.session.post(f"{self.api_url}/predict", json=payload, timeout=5)
            long_text_handled = response.status_code in [400, 422, 200]  # Puede ser manejado o procesado
        except Exception:
            long_text_handled = False
        
        # Test con payload inválido
        try:
            response = self.session.post(f"{self.api_url}/predict", json={"invalid": "payload"}, timeout=5)
            invalid_payload_handled = response.status_code in [400, 422]
        except Exception:
            invalid_payload_handled = False
        
        duration = time.time() - start_time
        
        if empty_text_handled and long_text_handled and invalid_payload_handled:
            self.log_test("Error Handling", True, "Errores manejados correctamente", duration)
            return True
        else:
            failed_tests = []
            if not empty_text_handled: failed_tests.append("empty_text")
            if not long_text_handled: failed_tests.append("long_text") 
            if not invalid_payload_handled: failed_tests.append("invalid_payload")
            self.log_test("Error Handling", False, f"Fallos: {failed_tests}", duration)
            return False
    
    def run_all_tests(self) -> Dict:
        """Ejecuta todos los tests"""
        print("🧪 Iniciando Test Suite para Interfaz Web")
        print("=" * 60)
        
        tests = [
            self.test_api_health,
            self.test_web_interface_loading,
            self.test_single_prediction,
            self.test_batch_prediction,
            self.test_probabilities_endpoint,
            self.test_model_info,
            self.test_web_static_files,
            self.test_performance_load,
            self.test_error_handling
        ]
        
        total_tests = len(tests)
        passed_tests = 0
        
        for test in tests:
            if test():
                passed_tests += 1
            time.sleep(0.5)  # Pausa entre tests
        
        print("\n" + "=" * 60)
        print(f"📊 RESUMEN DE TESTS")
        print(f"Total: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests/total_tests:.1%}")
        
        if passed_tests == total_tests:
            print("🎉 ¡TODOS LOS TESTS PASARON!")
        else:
            print("⚠️  Algunos tests fallaron. Revisar logs arriba.")
        
        return {
            "total": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests,
            "details": self.test_results
        }
    
    def generate_report(self, output_file: str = "test_report.json"):
        """Genera reporte detallado en JSON"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "api_url": self.api_url,
            "web_url": self.web_url,
            "summary": {
                "total_tests": len(self.test_results),
                "passed": sum(1 for r in self.test_results if r["passed"]),
                "failed": sum(1 for r in self.test_results if not r["passed"]),
                "success_rate": sum(1 for r in self.test_results if r["passed"]) / len(self.test_results) if self.test_results else 0
            },
            "test_details": self.test_results
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Reporte guardado en: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Test Suite para Interfaz Web del Transformer")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="URL de la API")
    parser.add_argument("--web-url", default="http://localhost:8080", help="URL de la interfaz web")
    parser.add_argument("--report", default="test_report.json", help="Archivo de reporte")
    parser.add_argument("--load-test", type=int, default=10, help="Número de requests para test de carga")
    
    args = parser.parse_args()
    
    # Crear suite de tests
    test_suite = WebInterfaceTestSuite(args.api_url, args.web_url)
    
    # Ejecutar tests
    results = test_suite.run_all_tests()
    
    # Generar reporte
    test_suite.generate_report(args.report)
    
    # Exit code según resultados
    exit_code = 0 if results["passed"] == results["total"] else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()