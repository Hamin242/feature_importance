import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.title('Feature importance')

st.sidebar.title('입력 값')

uploaded_file = st.sidebar.file_uploader('CSV 파일 업로드', type=['csv'])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file, encoding='EUC-KR')
    
    # Print column names to debug the column name issue
    st.write("Column names in the DataFrame:", df.columns)
    
    # Check if the column '양불' exists in the DataFrame
    if '양불' in df.columns:
        if df['양불'].dtype != int:
            df['양불'] = df['양불'].map({'불량': -1, '양품': 1})
        st.write(df)  # Display the DataFrame
    else:
        st.error("The column '양불' does not exist in the DataFrame.")

    # 반응 변수와 입력 변수를 선택할 수 있는 위젯을 추가합니다.
    target_column = st.sidebar.selectbox('반응 변수 선택', options=df.columns)
    input_columns = st.sidebar.multiselect('입력 변수 선택', options=df.columns, default=[])

    # 특징 변수와 타겟 변수를 선택된 컬럼으로 설정합니다.
    X = df[input_columns]  # 선택된 입력 변수로 설정합니다.
    y = df[target_column]  # 선택된 반응 변수로 설정합니다.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 탭 생성 : 첫번째 탭의 이름은 Tab A 로, Tab B로 표시합니다.
    tab1, tab2 = st.tabs(['SVM', 'Random Forest'])

    with tab1:
        # 모델 훈련 영역
        st.header('모델 훈련')
        if st.button('SVM 모델 학습'):
            svm_model = SVC(kernel='linear')  # 선형 SVM 모델을 생성합니다.
            svm_model.fit(X_train, y_train)  # SVM 모델을 학습시킵니다.
            st.subheader('SVM 모델 학습 완료')

            # SVM 모델 변수 중요도 출력
            st.write('SVM 모델 변수 중요도:')
            importance_sorted = sorted(zip(X.columns, svm_model.coef_[0]), key=lambda x: x[1], reverse=True)
            for feature, importance in importance_sorted:
                st.write(f'{feature}: {importance}')

            svm_importance_df = pd.DataFrame(importance_sorted, columns=['Feature', 'Importance'])
            st.download_button(
                label="Download Feature Importance",
                data=svm_importance_df.to_csv(index=False),
                file_name="svm_feature_importance.csv",
                mime="text/csv"
            )
            
            c = svm_model.intercept_[0]  # 직선의 절편
            st.subheader("직선 방정식:")
            tmp = f"y = "
            for feature, importance in importance_sorted:
                tmp += f" + {importance:.4f} * {feature}"
            tmp += f" + {c:.4f}"
            st.write(tmp)
            
            # 테스트 데이터에 대한 예측 및 정확도 출력
            predictions = svm_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            st.write(f'테스트 데이터 정확도: {accuracy:.4f}')

            st.header('결과 시각화')
            fig = plt.figure(figsize=(10, 6))
            plt.barh([x[0] for x in importance_sorted], [x[1] for x in importance_sorted], color='skyblue')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            plt.title('SVM Feature importance')
            st.pyplot(fig)



    with tab2:
        # 모델 훈련 영역
        st.header('모델 훈련')
        if st.button('Random Forest 모델 학습'):
            rf_model = RandomForestClassifier()  # 랜덤 포레스트 모델을 생성합니다.
            rf_model.fit(X_train, y_train)  # 랜덤 포레스트 모델을 학습시킵니다.
            st.subheader('Random Forest 모델 학습 완료')

            # Random Forest 모델 변수 중요도 출력
            st.write('Random Forest 모델 변수 중요도:')
            importance_sorted = sorted(zip(X.columns, rf_model.feature_importances_), key=lambda x: x[1], reverse=True)
            for feature, importance in importance_sorted:
                st.write(f'{feature}: {importance}')



            # 테스트 데이터에 대한 예측 및 정확도 출력
            predictions = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            st.write(f'테스트 데이터 정확도: {accuracy:.4f}')
            rf_importance_df = pd.DataFrame(importance_sorted, columns=['Feature', 'Importance'])
            st.download_button(
                label="Download Feature Importance",
                data=rf_importance_df.to_csv(index=False),
                file_name="rf_feature_importance.csv",
                mime="text/csv"
            )
            st.subheader('결과 시각화')
            fig1 = plt.figure(figsize=(10, 6))
            plt.barh([x[0] for x in importance_sorted], [x[1] for x in importance_sorted], color='skyblue')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            plt.title('Random Forest Feature importance')
            st.pyplot(fig1)
